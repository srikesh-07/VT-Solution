import argparse
import copy
import json
import math
import os
import random
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
import wandb
from torchmetrics.functional import f1_score
from torchvision import transforms
from tqdm import tqdm

# torch.use_deterministic_algorithms(True)
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

import LibMTL.weighting as weighting_methods
from data.dataset import VisualTaxanomyDataset
from data.sampler import CategoryWiseSampler
from data.utils import replace_nan, strat_shuffle_split
from LibMTL.args import get_weight_args
from model.classifier import FashionCLIPClassifier
from train.evaluator import VisualTaxonomyF1Metric
from train.loss_wrapper import LossWrapper

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            224,
            scale=(0.9, 1.0),
            ratio=(0.75, 1.3333),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            max_size=None,
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


class WeightedMethod:
    def __init__(
        self,
        config: dict,
        method,
        epochs: int,
        shared_params_func,
        zero_grad_share_params_func,
    ):
        self.methods = dict()
        for name, attrs in config.items():
            self.methods[name] = method().to("cuda")
            self.methods[name].task_num = len(attrs)
            self.methods[name].epochs = epochs
            self.methods[name].train_loss_buffer = np.zeros([len(attrs), epochs])
            self.methods[name].get_share_params = shared_params_func
            self.methods[name].zero_grad_share_params = shared_params_func
            self.methods[name].rep_grad = False
            self.methods[name].device = "cuda"
            self.methods[name].init_param()

    def set_epoch(self, epoch: int):
        for method in self.methods.values():
            method.epoch = epoch

    def set_loss_buffer(self, name: str, loss, epoch):
        self.methods[name].train_loss_buffer[:, epoch] = loss

    def parameters(self):
        params = list()
        for method in self.methods.values():
            for param in method.parameters():
                yield param

    def get_method(self, name: str):
        return self.methods[name]


class Trainer:
    def __init__(
        self,
        args: argparse.Namespace,
        label_config: Dict,
        device="cuda" if torch.cuda.device_count() else "cpu",
    ):
        self.args = args
        self.label_config = label_config

        self.img_dir = self.args.img_dir
        self.save_dir = self.args.save_dir
        self.csv_path = self.args.csv_path

        self.model = FashionCLIPClassifier(
            label_config,
            model_name=args.backbone,
            inter_layer=args.inter_linear,
            ckpt_path=args.pretrained_ckpt,
            batch_norm=args.batch_norm,
        )

        self.scorer = VisualTaxonomyF1Metric(self.label_config, device=device)
        self.num_categories = 5
        self.best_metrics = {"Loss": dict(), "Overall Scores": dict()}

        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        self.wandb = self.args.wandb
        self.exp_name = self.args.run_name + f"_block_{abs(self.args.unlock_block)}"
        self.save_dir = os.path.join(self.args.save_dir, self.args.run_name)
        self.device = device

        os.makedirs(self.save_dir, exist_ok=True)
        self.model.to(self.device)

        self.unblock_block = self.args.unlock_block

    def reset_metrics(self):
        self.metrics = {"Loss": dict()}
        self.scorer.reset()

    def calculate_loss(
        self,
        preds: List[torch.Tensor],
        gts: List[torch.Tensor],
        mask: List[torch.Tensor],
        loss_funcs: List[Callable] = None,
        discrete_return: bool = False,
        feats=None,
    ):
        loss = torch.tensor([0.0], dtype=torch.float16, device=self.device)
        if discrete_return:
            losses = list()
        if loss_funcs is None:
            loss_funcs = [None] * len(gts)
        assert len(mask) == len(gts) == len(loss_funcs)

        for pred, gt, m, loss_func in zip(preds, gts, mask, loss_funcs):
            m = m.bool().to(self.device)
            gt = gt.to(self.device)
            gt = gt[m]
            pred = pred[m, :]
            if gt.numel() > 0:
                if loss_func is None:
                    curr_loss = torch.nn.functional.cross_entropy(pred, gt)
                else:
                    if feats is None:
                        curr_loss = loss_func(pred, gt)
                    else:
                        curr_loss = loss_func(pred, gt, feats[m, :])
                if discrete_return:
                    losses.append(curr_loss)
                loss += curr_loss
            else:
                if discrete_return:
                    losses.append(torch.tensor(0.0, device="cuda"))
        if discrete_return:
            return loss, losses
        return loss

    def data_setup(self):
        df = pd.read_csv(self.csv_path)
        train_df, val_df = strat_shuffle_split(
            df, val_ratio=self.args.val_ratio, random_state=self.args.random_state
        )
        # train_df = pd.read_csv('./data_split/train_df.csv')
        # val_df = pd.read_csv('./data_split/val_df.csv')

        train_dataset = VisualTaxanomyDataset(
            data_df=train_df,
            img_dir=self.img_dir,
            attr_config=self.label_config,
            transform=train_transforms,
        )

        valid_dataset = VisualTaxanomyDataset(
            data_df=val_df,
            img_dir=self.img_dir,
            attr_config=self.label_config,
            transform=val_transforms,
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_sampler=CategoryWiseSampler(
                categorical_indices=train_dataset.all_categories_indices,
                mixed_batch_sampling=(
                    True if self.args.batch_sampling == "mixed" else False
                ),
                drop_last=True,
                shuffle=True,
                batch_size=self.batch_size,
            ),
            shuffle=False,
            pin_memory=True,
            num_workers=self.args.train_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_sampler=CategoryWiseSampler(
                categorical_indices=valid_dataset.all_categories_indices,
                mixed_batch_sampling=False,
                shuffle=False,
                drop_last=False,
                batch_size=self.batch_size,
            ),
            shuffle=False,
            pin_memory=True,
            num_workers=self.args.val_workers,
        )
        return train_loader, valid_loader

    def validate(self, model: torch.nn.Module, val_loader: torch.utils.data.DataLoader):
        loss = 0
        model.eval()
        with torch.no_grad():
            for imgs, attrs, cats, mask in tqdm(
                val_loader,
                desc="Performing Validation",
                total=len(val_loader.batch_sampler),
            ):
                out = model(x=imgs.to(self.device), category=cats[0])
                loss += self.calculate_loss(
                    preds=out, gts=attrs, loss_funcs=None, mask=mask
                )
                self.scorer.update(
                    preds=out, gts=attrs, cats=cats, mask=mask, verify=True
                )
        self.metrics["Loss"]["Val Loss"] = loss.detach().item() / len(
            val_loader.batch_sampler
        )
        self.metrics.update(self.scorer.compute(classwise_scores=True))

    def print_metrics(self, metrics: Dict):
        max_key_length = 40
        print("-" * max_key_length)
        print(
            f"{'Train Loss':<{max_key_length}} : {round(metrics['Loss'].pop('Train Loss'), 4)}"
        )
        print(
            f"{'Val Loss':<{max_key_length}} : {round(metrics['Loss'].pop('Val Loss'), 4)}"
        )
        print("-" * max_key_length)
        print(
            f"{'H.M F1-Score':<{max_key_length}} : {round(metrics['Overall Scores'].pop('H.M F1-Score'), 4)}"
        )
        print(
            f"{'Micro F1-Score':<{max_key_length}} : {round(metrics['Overall Scores'].pop('Micro F1-Score'), 4)}"
        )
        print(
            f"{'Macro F1-Score':<{max_key_length}} : {round(metrics['Overall Scores'].pop('Macro F1-Score'), 4)}"
        )
        print("-" * max_key_length)
        for key, value in metrics["Category-Wise Scores"].items():
            key = f"H.M {key} F1-Score"
            print(f"{key:<{max_key_length}} : {round(value['H.M F1-Score'], 4)}")
        print("-" * max_key_length)

    def save_ckpt(self, metrics: Dict, epoch_num: int, suffix: str = ""):
        path = os.path.join(self.save_dir, f"best_{suffix}.pt")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "metrics": metrics,
                "epoch": epoch_num,
                "args": vars(self.args),
            },
            path,
        )
        return path

    def lr_lambda(self, step: int):
        if step < self.warmup_steps:
            return step / self.warmup_steps  # Linear warmup
        else:
            progress = (step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay

    def _construct_params(self):
        params = [
            {
                "params": getattr(self.model, i).parameters(),
                "lr": self.args.scratch_lr,
                "weight_decay": self.args.scratch_weight_decay,
            }
            for i in self.model.mapping.values()
        ]
        if self.args.batch_norm:
            params.append(
                {
                    "params": getattr(self.model, "b_norm").parameters(),
                    "lr": self.args.scratch_lr,
                    "weight_decay": 0,
                }
            )
        if self.args.inter_linear:
            params.append(
                {
                    "params": getattr(self.model, "inter_linear").parameters(),
                    "lr": self.args.scratch_lr,
                    "weight_decay": self.args.scratch_weight_decay,
                }
            )

        if self.args.unfreeze_proj:
            self.model.clip_model.visual.proj.requires_grad = True
            params.append(
                {
                    "params": self.model.clip_model.visual.proj,
                    "lr": self.args.pretrained_lr,
                    "weight_decay": self.args.pretrained_weight_decay,
                }
            )
        if self.args.unfreeze_cls:
            self.model.clip_model.visual.class_embedding.requires_grad = True
            params.append(
                {
                    "params": self.model.clip_model.visual.class_embedding,
                    "lr": self.args.pretrained_lr,
                    "weight_decay": self.args.pretrained_weight_decay,
                }
            )
        if self.args.unfreeze_pos:
            self.model.clip_model.visual.positional_embedding.requires_grad = True
            params.append(
                {
                    "params": self.model.clip_model.visual.positional_embedding,
                    "lr": self.args.pretrained_lr,
                    "weight_decay": 0.0,
                }
            )
        if self.args.unfreeze_top:
            self.model.clip_model.visual.conv1.requires_grad = True
            self.model.clip_model.visual.ln_pre.requires_grad = True
            params.append(
                {
                    "params": self.model.clip_model.visual.conv1.parameters(),
                    "lr": self.args.pretrained_lr,
                    "weight_decay": self.args.pretrained_weight_decay,
                }
            )
            params.append(
                {
                    "params": self.model.clip_model.visual.ln_pre.parameters(),
                    "lr": self.args.pretrained_lr,
                    "weight_decay": 0.0,
                }
            )
        decay_params = list()
        ndecay_params = list()
        for name, param in self.model.clip_model.visual.transformer.resblocks[
            self.unblock_block :
        ].named_parameters():
            if name.startswith("ln"):
                param.requires_grad = True
                ndecay_params.append(param)
            else:
                param.requires_grad = True
                decay_params.append(param)
        params.append(
            {
                "params": decay_params,
                "lr": self.args.pretrained_lr,
                "weight_decay": self.args.pretrained_weight_decay,
            }
        )
        params.append(
            {
                "params": ndecay_params,
                "lr": self.args.pretrained_lr,
                "weight_decay": 0.0,
            }
        )
        self.model.clip_model.visual.ln_post.requires_grad = True
        params.append(
            {
                "params": self.model.clip_model.visual.ln_post.parameters(),
                "lr": self.args.pretrained_lr,
                "weight_decay": 0,
            }
        )

        return params

    def save_metrics(self, prefix: str = ""):
        with open(
            os.path.join(self.save_dir, f"{prefix}_detailed_metrics.json"), "w"
        ) as j_file:
            json.dump(self.metrics, j_file, indent=4)

    def train(self):
        train_loader, val_loader = self.data_setup()
        loss_wrapper = LossWrapper(
            cls_num_dict=train_loader.dataset.class_count(),
            args=self.args,
            device=self.device,
        )
        params = self._construct_params()

        if self.args.method:
            w_method = weighting_methods.__dict__[self.args.method]
            weighted_method = WeightedMethod(
                config=self.label_config,
                method=w_method,
                epochs=self.args.epochs,
                shared_params_func=self.model.get_share_params,
                zero_grad_share_params_func=self.model.zero_grad_share_params,
            )
            params.append(
                {
                    "params": weighted_method.parameters(),
                    "lr": 1e-4,
                    "weight_decay": 1e-5,
                }
            )
            weight_args = get_weight_args(self.args.method)

        optimizer = torch.optim.AdamW(params)

        self.total_steps = len(train_loader.batch_sampler) * self.epochs
        self.warmup_steps = int(0.1 * self.total_steps)  # 10% of total steps for warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self.lr_lambda
        )

        if self.wandb:
            wandb.init(name=self.exp_name, project=self.args.wandb_project)

        patience = 0
        best_loss = float("inf")
        best_acc = 0

        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            loss_dict = {name: list() for name in self.label_config.keys()}
            if self.args.method:
                weighted_method.set_epoch(epoch - 1)
            pbar = tqdm(
                train_loader,
                desc="Performing Training",
                total=len(train_loader.batch_sampler),
            )
            print(f"\nEpoch - {epoch} / {self.epochs}")
            self.reset_metrics()
            self.model.train()
            for imgs, attrs, cats, mask in pbar:
                for cat in cats:
                    assert cat == cats[0]
                feats = None
                if (
                    self.args.loss_type.startswith("IB")
                    and epoch >= loss_wrapper.start_ib_epoch
                ):
                    out, feats = self.model(
                        x=imgs.to(self.device), category=cats[0], return_emb=True
                    )
                else:
                    out = self.model(x=imgs.to(self.device), category=cats[0])
                loss, losses = self.calculate_loss(
                    preds=out,
                    gts=attrs,
                    mask=mask,
                    loss_funcs=loss_wrapper(cats[0], epoch),
                    discrete_return=True,
                    feats=feats,
                )
                loss_dict[cats[0]].append([l.detach().item() for l in losses])
                optimizer.zero_grad(set_to_none=False)
                if self.args.method is None:
                    loss.backward()
                else:
                    weighted_method.get_method(cats[0]).backward(
                        torch.stack(losses), **weight_args
                    )
                total_loss += loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                torch.cuda.empty_cache()
                pbar.set_postfix_str(
                    f"loss={round(float(loss), 4)} lr={optimizer.param_groups[0]['lr']}"
                )
            for name, values in loss_dict.items():
                arr = np.array(values)
                m_arr = np.mean(arr, axis=0)
                if self.args.method:
                    weighted_method.set_loss_buffer(
                        name=name, epoch=epoch - 1, loss=m_arr
                    )
            self.validate(model=self.model, val_loader=val_loader)
            self.metrics["Loss"]["Train Loss"] = total_loss.item() / len(
                train_loader.batch_sampler
            )
            if self.best_metrics["Loss"].get("Val Loss") is not None and (
                self.best_metrics["Loss"]["Val Loss"] < self.metrics["Loss"]["Val Loss"]
            ):
                patience += 1
                print(f"[INFO] Patience - {patience}")
            else:
                patience = 0

            if self.best_metrics["Overall Scores"].get("H.M F1-Score") is None or (
                (
                    self.best_metrics["Overall Scores"]["H.M F1-Score"]
                    < self.metrics["Overall Scores"]["H.M F1-Score"]
                )
                or (
                    self.best_metrics["Loss"]["Val Loss"]
                    >= self.metrics["Loss"]["Val Loss"]
                )
            ):
                if self.best_metrics["Overall Scores"].get("H.M F1-Score") is None or (
                    (
                        self.best_metrics["Overall Scores"]["H.M F1-Score"]
                        < self.metrics["Overall Scores"]["H.M F1-Score"]
                    )
                    and (
                        self.best_metrics["Loss"]["Val Loss"]
                        >= self.metrics["Loss"]["Val Loss"]
                    )
                ):
                    self.best_metrics = copy.deepcopy(self.metrics)
                    self.save_ckpt(
                        metrics=self.best_metrics, epoch_num=epoch, suffix="both"
                    )
                    self.save_metrics(prefix="both")
                    print(
                        "[INFO] Best of All Checkpoint and its Detailed Metrics have been saved"
                    )
                elif (
                    self.best_metrics["Overall Scores"]["H.M F1-Score"]
                    < self.metrics["Overall Scores"]["H.M F1-Score"]
                    and best_acc < self.metrics["Overall Scores"]["H.M F1-Score"]
                ):
                    best_acc = self.metrics["Overall Scores"]["H.M F1-Score"]
                    self.save_ckpt(
                        metrics=self.metrics, epoch_num=epoch, suffix="score"
                    )
                    self.save_metrics(prefix="score")
                    print(
                        "[INFO] Best Score Checkpoint and its Detailed Metrics have been saved"
                    )
                elif (
                    self.best_metrics["Loss"]["Val Loss"]
                    >= self.metrics["Loss"]["Val Loss"]
                    and best_loss > self.metrics["Loss"]["Val Loss"]
                ):
                    best_loss = self.metrics["Loss"]["Val Loss"]
                    self.save_ckpt(metrics=self.metrics, epoch_num=epoch, suffix="loss")
                    self.save_metrics(prefix="loss")
                    print(
                        "[INFO] Best Loss Checkpoint and its Detailed Metrics have been saved"
                    )

            self.print_metrics(self.metrics.copy())

            if self.wandb:
                self.metrics.update({"Learning Rate": optimizer.param_groups[0]["lr"]})
                del self.metrics["Attribute-Wise Scores"]
                wandb.log(self.metrics)

            if patience == self.args.patience:
                print(
                    f"Early stopping due to no improvement for {self.args.patience}  Epochs"
                )
                print("Best Loss: ", round(best_loss, 4))
                print("Best F1-Score: ", round(best_acc, 4))
                print("Overall Best: ")
                print(json.dumps(self.best_metrics, indent=4))
                break

            torch.cuda.empty_cache()
