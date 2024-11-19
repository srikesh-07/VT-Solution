import os
import traceback

import pandas as pd
import torch
from PIL import Image
from torchvision.datasets import VisionDataset


class VisualTaxanomyDataset(VisionDataset):
    """
    Module Name: VisualTaxanomyDataset
    Description: A class of Torchvision's `VisionDataset` that is made for the multi-attribute prediction dataset.
    
    Author:
        - Srikeshram B (srikeshram05@gmail.com)
    
    Attributes:
        - data_df (pd.DataFrame): Dataframe contains the ground truth of the Dataset.
        - img_dir (str): Path to the image directory containing images in DataFrame.
        - attr_config (dict): Dictionary contains information regarding Categories, Attributes, and their respective Class Names.
        - transform (torchvision.Transforms): Contains the list of transformations to be applied to the images. Defaults to None.
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        img_dir: str,
        attr_config: dict,
        transform=None,
    ):
        """
        Initializes the Visual Taxonomy Datases.
        
        Args:
            - data_df (pd.DataFrame): Dataframe contains the ground truth of the Dataset.
            - img_dir (str): Path to the image directory containing images in DataFrame.
            - attr_config (dict): Dictionary contains information regarding Categories, Attributes, and their respective Class Names.
            - transform (torchvision.Transforms): Contains the list of transformations to be applied to the images. Defaults to None.
        """
        super().__init__(transform=transform, root=None)
        self.data_df = data_df
        self.attr_config = attr_config
        self.img_dir = img_dir

    def class_count(self):
        """
        Calculates the Class Count for each attribute for each category.

        Returns:
            - cls_count (dict): A Dictionary of Nested List that contains the class counts of each attribute in the respective category.
        """
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
        """
        Calculates the Category Count.

        Returns:
            - cls_count (dict): A Dictionary with an integer value contains the instance count of each category.
        """
        cat_count = list()
        for cat in self.attr_config.keys():
            cat_df = self.data_df[self.data_df["Category"] == cat]
            cat_count.append(len(cat_df))
        return cat_count

    def __getitem__(self, idx: int):
        """
        Returns the Transformed Image with the List of Ground-Truth Labels and Category and Mask to identify the null values.

        Returns:
            - img (torch.Tensor): A Tensor that is transformed as a Tensor.
            - labels (list[int]): A list of integer values represents the class of each attribute.
            - category (str): String of category name.
            - mask (list): Boolean list which contains 0 that represent the labels at the specific index is null.
        """
        row = self.data_df.iloc[idx]
        assert row["len"] == len(self.attr_config[row["Category"]])
        mask = list()
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
        img = Image.open(os.path.join(self.img_dir, f"{row['id']:06d}.jpg"))
        if self.transform:
            img = self.transform(img)
        return img, labels, row["Category"], mask

    def __len__(self):
        """
        Returns the length of the Dataset
        """
        return len(self.data_df)

    @property
    def all_categories_indices(self):
        """
        Returns the list of indices of the DataFrame for the each category.

        Returns:
         out (dict: List): Dictionary with the list of indices of the DataFrame for the each category
        """
        out = dict()
        for category in self.attr_config.keys():
            out[category] = self.data_df[
                self.data_df["Category"] == category
            ].index.to_list()
        return out

    @property
    def class_config(self):
         """
        Returns the attribute config.
        """
        return self.attr_config
