import argparse
import os

import LibMTL.weighting as methods
from train.loss_wrapper import LossWrapper


def get_train_args():
    parser = argparse.ArgumentParser(
        prog="Visual Taxonomy Challenge - Training Script",
        description="Training Script to train the model for predicting the attributes of the clothes.",
    )

    # File related settings
    parser.add_argument(
        "--img_dir",
        required=True,
        help="Directory path that contains the image files for Training",
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="CSV file path that contains the ground-truth for Training",
    )
    parser.add_argument(
        "--attributes_config",
        default=os.path.join("config", "attrs_config.json"),
        help="JSON file path that contains the attributes and its list of labels for each catrgory.",
    )

    # Output Settings
    parser.add_argument(
        "--save_dir", default="output", help="Directory path to store the checkpoints."
    )
    parser.add_argument(
        "--run_name",
        default="visual_taxonomy_train",
        help="Directory path to store the checkpoints.",
    )

    # WandB Related Settings
    parser.add_argument(
        "--wandb",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable WandB Integration. Defaults to True.",
    )
    parser.add_argument(
        "--wandb_project",
        default="Visual Taxonomy - Training",
        help="Name of the WandB project.",
    )

    # Model Related Settings
    parser.add_argument(
        "--backbone",
        default="ViT-B-32",
        choices=["ViT-B-32", "ViT-L-14"],
        help="Type of ViT model to be used as Backbone.",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        default="weights/marqo-gcl-vitb32-127-gs-full_states.pt",
        help="Pre-Trained weights path for the backbone.",
    )
    parser.add_argument(
        "--unlock_block",
        type=int,
        default=-8,
        help="Negative integer which describes the number of ResBlocks from bottom to unfreeze for Training.",
    )
    parser.add_argument(
        "--inter_linear",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Intermediate Linear Layer after encoder of the pre-trained model.",
    )
    parser.add_argument(
        "--batch_norm",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Batch Norm Layer after encoder of the pre-trained model and before inter-linear.",
    )
    parser.add_argument(
        "--unfreeze_proj",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Unfreeze Projection Layer for Training.",
    )
    parser.add_argument(
        "--unfreeze_cls",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Unfreeze Class Embedding Layer for Training.",
    )
    parser.add_argument(
        "--unfreeze_pos",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Unfreeze Positional Embedding Layer for Training.",
    )
    parser.add_argument(
        "--unfreeze_top",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Unfreeze Top Conv Layer and Top Norm Layer for Training.",
    )

    # Training Settings
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation ratio for the train-test stratified shuffle split.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs for training."
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch Size for Training."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for the Early Stopping to stop training.",
    )
    parser.add_argument(
        "--train_workers",
        type=int,
        default=5,
        help="Number of Workers for train Data Loaders.",
    )
    parser.add_argument(
        "--val_workers",
        type=int,
        default=2,
        help="Number of Workers for validation Data Loaders.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random State for the Train-Test Data Split.",
    )
    parser.add_argument(
        "--batch_sampling",
        default="mixed",
        choices=["mixed", "sequential"],
        help="Type of Batch-Sampling for the Training which is either sequential or mixed.",
    )

    parser.add_argument(
        "--method",
        default=None,
        choices=methods.__all__,
        help="Type of Loss Weighting Method.",
    )

    parser.add_argument(
        "--train_rule",
        default="None",
        choices=LossWrapper.rules(),
        help="Type of Loss Rules Method.",
    )

    parser.add_argument(
        "--loss_type",
        default="CE",
        choices=LossWrapper.losses(),
        help="Type of Loss Type Method.",
    )

    # Hyper-Parameter Settings
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="Gamma for the Focal Loss."
    )
    parser.add_argument(
        "--beta", type=float, default=0.999, help="Beta for the CBL Loss and DRW."
    )
    parser.add_argument(
        "--pretrained_lr",
        type=float,
        default=1e-5,
        help="LR for the unforzen pre-trained layers of the Backbone.",
    )
    parser.add_argument(
        "--pretrained_weight_decay",
        type=float,
        default=1e-4,
        help="Weight Decay for the unforzen pre-trained layers of the Backbone.",
    )
    parser.add_argument(
        "--scratch_lr",
        type=float,
        default=1e-4,
        help="LR for the FC layers to be trained from Scratch.",
    )
    parser.add_argument(
        "--scratch_weight_decay",
        type=float,
        default=1e-4,
        help="Weight Decay for the FC layers to be trained from Scratch.",
    )

    args = parser.parse_args()

    return args


def get_inference_args():
    parser = argparse.ArgumentParser(
        prog="Visual Taxonomy Challenge - Inference Script",
        description="Inference Script to inference the model for predicting the attributes of the clothes.",
    )

    # File related settings
    parser.add_argument(
        "--img_dir",
        required=True,
        help="Directory path that contains the image files for Training",
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="CSV file path that contains the ground-truth for Training",
    )
    parser.add_argument(
        "--ckpt_path", required=True, help="Checkpoint file path to be inferenced."
    )
    parser.add_argument(
        "--attributes_config",
        default=os.path.join("config", "attrs_config.json"),
        help="JSON file path that contains the attributes and its list of labels for each catrgory.",
    )
    parser.add_argument(
        "--out_csv_path",
        default=None,
        help="Output file path with filename to store the inference as CSV. By default it stores it in the same folder as checkpoints.",
    )

    args = parser.parse_args()

    return args
