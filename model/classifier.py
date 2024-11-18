import os

import open_clip
import torch

from model.layers import ConditionalBatchNorm, ConditionalLinear, VisualEncoder


class FashionCLIPClassifier(torch.nn.Module):
    def __init__(
        self,
        config: dict,
        model_name: str = "ViT-B-32",
        ckpt_path: str = "weights/gcl-vitl14-124-gs-full-states.pt",
        inter_layer: bool = False,
        batch_norm: bool = False,
        normalize_enc_out: bool = False,
    ):
        super().__init__()
        out_features = 512
        self.clip_model = VisualEncoder(name=model_name, pretrained_path=ckpt_path)
        self.inter_layer = inter_layer
        self.normalize_enc_out = normalize_enc_out
        self.cats = list(config.keys())

        if batch_norm:
            self.b_norm = ConditionalBatchNorm(len(self.cats), out_features)
        else:
            self.b_norm = None

        if self.inter_layer:
            self.inter_linear = ConditionalLinear(
                len(self.cats), out_features, out_features // 2
            )
            self.activation = torch.nn.ReLU()
            out_features = out_features // 2
        else:
            self.inter_layer = None

        self.mapping = dict()
        for key, attrs in config.items():
            self.mapping[key] = f"{key.lower()}_fcs"
            fc = torch.nn.ModuleList()
            for val in attrs.values():
                fc.append(torch.nn.Linear(out_features, len(val)))
            setattr(self, self.mapping[key], fc)

        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor, category: str, return_emb: bool = False):
        if self.mapping.get(category, None) is None:
            raise ValueError(f"Invalid Category - {category}")
        cat_idx = self.cats.index(category)
        x = self.clip_model(x)
        if self.b_norm:
            x = self.b_norm(x, category_idx=cat_idx)
        if self.inter_layer:
            feat = self.inter_linear(x, cat_idx)
            x = self.activation(feat)
        out = list()
        for layer in getattr(self, self.mapping[category]):
            out.append(layer(x))
        if return_emb:
            return out, torch.sum(torch.abs(feat), 1).reshape(-1, 1)
        return out

    def shared_param_generator(self):
        for param in self.clip_model.parameters():
            if param.requires_grad:
                yield param

    def get_share_params(self):
        return self.shared_param_generator()

    def zero_grad_share_params(self):
        self.clip_model.zero_grad(set_to_none=False)

    def task_specific_parameters(self, name: str):
        return list(getattr(self, self.mapping[name]).parameters())

    def last_shared_parameters(self, name: int):
        params = list()
        idx = self.cats.index(name)
        if self.inter_layer:
            for param in self.inter_linear.ln_layers[idx].parameters():
                params.append(param)
        return params
