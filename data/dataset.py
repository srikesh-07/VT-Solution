import os
import traceback

import pandas as pd
import torch
from PIL import Image
from torchvision.datasets import VisionDataset


class VisualTaxanomyDataset(VisionDataset):
    def __init__(
        self,
        data_df: pd.DataFrame,
        img_dir: str,
        attr_config: dict,
        transform=None,
    ):
        super().__init__(transform=transform, root=None)
        self.data_df = data_df
        self.attr_config = attr_config
        self.img_dir = img_dir

    def class_count(self):
        cls_count = dict()
        for cat, attrs in self.attr_config.items():
            cat_df = self.data_df[self.data_df["Category"] == cat]
            cls_count[cat] = list()
            for idx, classes in enumerate(attrs.values(), start=1):
                counts = cat_df[f"attr_{idx}"].value_counts()
                counts = [counts[cls] for cls in classes]
                cls_count[cat].append(counts)
        return cls_count

    def category_count(self):
        cat_count = list()
        for cat in self.attr_config.keys():
            cat_df = self.data_df[self.data_df["Category"] == cat]
            cat_count.append(len(cat_df))
        return cat_count

    def __getitem__(self, idx: int):
        row = self.data_df.iloc[idx]
        assert row["len"] == len(self.attr_config[row["Category"]])
        mask = list()
        try:
            labels = list()
            for i, cls_vls in zip(
                range(1, row["len"] + 1), self.attr_config[row["Category"]].values()
            ):
                if not isinstance(row[f"attr_{i}"], str):
                    labels.append(-1)
                    mask.append(0)
                else:
                    labels.append(cls_vls.index(row[f"attr_{i}"]))
                    mask.append(1)
        except Exception as e:
            print(traceback.format_exc(e))
            print(row)
            exit(0)
        img = Image.open(os.path.join(self.img_dir, f"{row['id']:06d}.jpg"))
        if self.transform:
            img = self.transform(img)
        return img, labels, row["Category"], mask

    def __len__(self):
        return len(self.data_df)

    @property
    def all_categories_indices(self):
        out = dict()
        for category in self.attr_config.keys():
            out[category] = self.data_df[
                self.data_df["Category"] == category
            ].index.to_list()
        return out

    @property
    def class_config(self):
        return self.attr_config
