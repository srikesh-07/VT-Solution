import argparse
import json
import os
from timeit import default_timer as timer
from typing import Dict

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from model.classifier import FashionCLIPClassifier
from train.trainer import val_transforms
from utils.params import get_inference_args


def load_model(ckpt_path: str, attr_config: Dict, args: argparse.Namespace):
    ckpt_data = torch.load(ckpt_path)
    if ckpt_data.get("args", None) is None:
        print("[INFO] No config found for the model.")
        print("[INFO] Defaulting the inter_linear and batch_norm to True")
        model = FashionCLIPClassifier(
            attr_config,
            inter_layer=args.inter_linear,
            ckpt_path=None,
            batch_norm=args.batch_norm,
        )
    else:
        model = FashionCLIPClassifier(
            attr_config,
            inter_layer=ckpt_data["args"]["inter_linear"],
            ckpt_path=None,
            batch_norm=ckpt_data["args"]["batch_norm"],
        )
    model.load_state_dict(ckpt_data["state_dict"], strict=True)
    model.eval()

    return model


def perform_inference(
    ckpt_path: str,
    args: argparse.Namespace,
    attr_config: Dict,
    out_csv_path: str = None,
):
    model = load_model(ckpt_path=ckpt_path, attr_config=attr_config, args=args)
    out = dict(id=list(), Category=list(), len=list())
    for i in range(1, 11):
        out[f"attr_{i}"] = list()

    df = pd.read_csv(args.csv_path)

    timer_dict = dict()
    for name in attr_config.keys():
        timer_dict[name] = [0, 0]

    with torch.no_grad():
        for idx in tqdm(df.index):
            start = timer()
            row = df.iloc[idx]
            out["id"].append(row["id"])
            out["Category"].append(row["Category"])
            img = val_transforms(
                Image.open(os.path.join(args.img_dir, f"{row['id']:06d}.jpg"))
            ).unsqueeze(0)
            out_m = model(img, row["Category"])
            attr_idx = 1
            out["len"].append(len(out_m))
            for attr, classes in zip(out_m, attr_config[row["Category"]].values()):
                max_idx = torch.argmax(attr[0])
                out[f"attr_{attr_idx}"].append(classes[int(max_idx.item())])
                attr_idx += 1
            for i in range(attr_idx, 11):
                out[f"attr_{i}"].append("dummy_value")
            end = timer()
            timer_dict[row["Category"]][0] += round((end - start), 4)
            timer_dict[row["Category"]][1] += 1

    if out_csv_path is None:
        out_csv_path = f"{os.path.splitext(ckpt_path)[0]}.csv"

    out_df = pd.DataFrame.from_dict(out)
    out_df.to_csv(out_csv_path, header=True, index=False)

    print(f"Average Timing Benchmark for Processing the - {args.ckpt_path}")
    print("-" * 30)
    for key, value in timer_dict.items():
        print(f"{key:<{30}} : {value[0] / value[1]} sec/img")
    print("-" * 30)


if __name__ == "__main__":
    args = get_inference_args()
    with open(args.attributes_config) as j_file:
        labels_config = json.load(j_file)
    if os.path.isdir(args.ckpt_path):
        if args.out_csv_path:
            assert os.path.isdir(
                args.out_csv_path
            ), "The output CSV path should be a valid folder path if the ckpt_path is a folder."
        for name in os.listdir(args.ckpt_path):
            print(f"[INFO] Performing Inference with {name}")
            if not name.endswith(".pt"):
                continue
            perform_inference(
                ckpt_path=os.path.join(args.ckpt_path, name),
                args=args,
                attr_config=labels_config,
                out_csv_path=(
                    os.path.join(args.out_csv_path, f"{os.path.splitext(name)[0]}.csv")
                    if args.out_csv_path
                    else None
                ),
            )
    else:
        if args.out_csv_path:
            assert args.out_csv_path.endswith(
                ".csv"
            ), "The output CSV path should be a valid path endswith .csv"
        perform_inference(
            ckpt_path=args.ckpt_path,
            args=args,
            attr_config=labels_config,
            out_csv_path=args.out_csv_path,
        )
