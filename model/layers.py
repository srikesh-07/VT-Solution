import open_clip
import torch


class ConditionalBatchNorm(torch.nn.Module):
    def __init__(self, num_categories, num_features):
        super(ConditionalBatchNorm, self).__init__()
        self.bn_layers = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features) for _ in range(num_categories)]
        )

    def forward(self, x, category_idx):
        return self.bn_layers[category_idx](x)


class ConditionalLinear(torch.nn.Module):
    def __init__(self, num_categories, in_features, out_features):
        super(ConditionalLinear, self).__init__()
        self.ln_layers = torch.nn.ModuleList(
            [torch.nn.Linear(in_features, out_features) for _ in range(num_categories)]
        )
        self.dropout = torch.nn.Dropout(0.4)

    def forward(self, x, category_idx):
        return self.dropout(self.ln_layers[category_idx](x))


class VisualEncoder(torch.nn.Module):
    def __init__(
        self,
        name="ViT-B-32",
        pretrained_path="./weights/marqo-gcl-vitb32-127-gs-full_states.pt",
    ):
        super().__init__()
        model, self.train_transforms, self.val_transforms = (
            open_clip.create_model_and_transforms(name, pretrained_path)
        )
        self.visual = model.visual

    def forward(self, x):
        return self.visual(x)
