import json
import os

import torch
import wandb

from train.trainer import Trainer
from utils.params import get_train_args

if __name__ == "__main__":
    args = get_train_args()
    print(args)
    with open(args.attributes_config, "r") as j_file:
        attrs_val_dict = json.load(j_file)
    print(f"Performing Training with Unfrozen {args.unlock_block} Blocks")
    a = Trainer(label_config=attrs_val_dict, args=args)
    a.train()
    if wandb.run is not None:
        wandb.finish()
    assert wandb.run is None
