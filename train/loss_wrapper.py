import argparse

import numpy as np
import torch

from train.default_loss import ClassBalancedLoss
from train.default_loss import FocalLoss as DefaultFocalLoss
from train.other_loss import FocalLoss, IB_FocalLoss, IBLoss, LDAMLoss, VSLoss


class LossWrapper:
    def _calculate_weights(self, epoch: int, cls_num_list: list):
        if self.train_rule == "CBReweight":
            beta = 0.999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
        elif self.train_rule == "IBReweight":
            per_cls_weights = 1.0 / np.array(cls_num_list)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
        elif self.train_rule == "DRW":
            idx = epoch // self.dividing_factor
            if epoch < 5:
                betas = [0]
                idx = 0
            else:
                slope = self.beta / epoch
                betas = [slope * epoch]
                idx = 0
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)
        else:
            per_cls_weights = None

        return per_cls_weights

    def _init_loss(self, cls_num_list: list, per_cls_weights: torch.Tensor = None):
        criterion_ib = None
        if self.loss_type == "CE":
            criterion = torch.nn.CrossEntropyLoss(weight=per_cls_weights).to(
                self.device
            )
        elif self.loss_type == "LDAM":
            criterion = LDAMLoss(
                cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights
            ).to(self.device)
        elif self.loss_type == "Focal":
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).to(self.device)
        elif self.loss_type == "IB":
            criterion = torch.nn.CrossEntropyLoss(weight=None).to(self.device)
            criterion_ib = IBLoss(
                weight=per_cls_weights, num_classes=len(cls_num_list), alpha=1000
            ).to(self.device)
        elif self.loss_type == "IBFocal":
            criterion = torch.nn.CrossEntropyLoss(weight=None).to(self.device)
            criterion_ib = IB_FocalLoss(
                weight=per_cls_weights,
                num_classes=len(cls_num_list),
                alpha=1000,
                gamma=1,
            ).to(self.device)
        elif self.loss_type == "F-CB":
            criterion = DefaultFocalLoss(
                num_classes=len(cls_num_list), reduction="none"
            ).to(self.device)
            if per_cls_weights is not None:
                criterion = ClassBalancedLoss(
                    loss_func=criterion, weight=per_cls_weights
                ).to(self.device)
            else:
                criterion = ClassBalancedLoss(
                    loss_func=criterion, samples_per_cls=cls_num_list
                ).to(self.device)
        elif self.loss_type == "CE-CB":
            criterion = torch.nn.CrossEntropyLoss(reduction="none").to(self.device)
            if per_cls_weights is not None:
                criterion = ClassBalancedLoss(
                    loss_func=criterion, weight=per_cls_weights
                ).to(self.device)
            else:
                criterion = ClassBalancedLoss(
                    loss_func=criterion, samples_per_cls=cls_num_list
                ).to(self.device)
        elif self.loss_type == "LA":
            criterion = VSLoss(
                cls_num_list=cls_num_list, tau=1.0, gamma=0, weight=per_cls_weights
            ).to(self.device)
        elif self.loss_type == "CDT":
            criterion = VSLoss(
                cls_num_list=cls_num_list, tau=0, gamma=0.3, weight=per_cls_weights
            ).to(self.device)
        elif self.loss_type == "VS":
            criterion = VSLoss(
                cls_num_list=cls_num_list, tau=1.0, gamma=0.3, weight=per_cls_weights
            ).to(self.device)
        else:
            raise ValueError(f"Invalid Loss Type - {self.loss_type}")
        return criterion, criterion_ib

    def __init__(
        self, cls_num_dict: dict, args: argparse.Namespace, device: str = "cpu"
    ):
        self.loss = dict()
        self.train_rule = args.train_rule
        self.loss_type = args.loss_type
        self.cls_num_dict = cls_num_dict
        self.dividing_factor = 20
        self.device = device

        self.beta = args.beta

        if args.loss_type == "IB" or args.loss_type == "IBFocal":
            self.start_ib_epoch = args.epochs // 2
        else:
            self.start_ib_epoch = None
        self.start_ib_epoch = 15

        for name, attrs in cls_num_dict.items():
            losses = list()
            ib_losses = list()
            for cats in attrs:
                if self.train_rule != "DRW":
                    cls_weights = self._calculate_weights(None, cats)
                    l, i_l = self._init_loss(
                        cls_num_list=cats, per_cls_weights=cls_weights
                    )
                    losses.append(l)
                    ib_losses.append(i_l)
                else:
                    l, i_l = self._init_loss(cls_num_list=cats)
                    losses.append(l)
                    ib_losses.append(i_l)
            self.loss[name] = (losses, ib_losses)

    @staticmethod
    def rules():
        return ["None", "CBReweight", "IBReweight", "DRW"]

    @staticmethod
    def losses():
        return [
            "CE",
            "LDAM",
            "Focal",
            "IB",
            "IBFocal",
            "F-CB",
            "CE-CB",
            "LA",
            "CDT",
            "VS",
        ]

    def __call__(self, category: str, epoch: int):
        losses, ib_losses = self.loss[category]
        if ib_losses[0] is None or epoch < self.start_ib_epoch:
            if self.train_rule == "DRW" and not self.loss_type.startswith("IB"):
                for loss, cls_num in zip(losses, self.cls_num_dict[category]):
                    loss.weight = self._calculate_weights(
                        epoch=epoch, cls_num_list=cls_num
                    )
            return losses
        else:
            if self.train_rule == "DRW":
                for loss, cls_num in zip(ib_losses, self.cls_num_dict[category]):
                    loss.weight = self._calculate_weights(
                        epoch=epoch, cls_num_list=cls_num
                    )
            return ib_losses
