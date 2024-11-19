# Training Arguments

- `--img_dir` **IMG_DIR**  
  Directory path that contains the image files for Training.

- `--csv_path` **CSV_PATH**  
  CSV file path that contains the ground truth for Training.

- `--attributes_config` **ATTRIBUTES_CONFIG**  
  JSON file path that contains the attributes and their list of labels for each category. Defaults to `config/attrs_config.json`

- `--save_dir` **SAVE_DIR**  
  Directory path to store the checkpoints.

- `--run_name` **RUN_NAME**  
  Name of the run (used for organizing checkpoints).

- `--wandb, --no-wandb`  
  Enable WandB Integration. Defaults to `True`.

- `--wandb_project` **WANDB_PROJECT**  
  Name of the WandB project.

- `--backbone` **{ViT-B-32,ViT-L-14}**  
  Type of ViT model to be used as Backbone.

- `--pretrained_ckpt` **PRETRAINED_CKPT**  
  Pre-trained weights path for the backbone.

- `--unlock_block` **UNLOCK_BLOCK**  
  Negative integer describing the number of ResBlocks (from bottom) to unfreeze for Training.

- `--inter_linear, --no-inter_linear`  
  Intermediate Linear Layer after the encoder of the pre-trained model. Defaults to `True`.

- `--batch_norm, --no-batch_norm`  
  Batch Norm Layer after the encoder of the pre-trained model and before the inter-linear layer. Defaults to `True`.

- `--unfreeze_proj, --no-unfreeze_proj`  
  Unfreeze Projection Layer for Training. Defaults to `True`.

- `--unfreeze_cls, --no-unfreeze_cls`  
  Unfreeze Class Embedding Layer for Training. Defaults to `False`.

- `--unfreeze_pos, --no-unfreeze_pos`  
  Unfreeze Positional Embedding Layer for Training. Defaults to `False`.

- `--unfreeze_top, --no-unfreeze_top`  
  Unfreeze Top Conv Layer and Top Norm Layer for Training. Defaults to `False`.

- `--val_ratio` **VAL_RATIO**  
  Validation ratio for the train-test stratified shuffle split.

- `--epochs` **EPOCHS**  
  Number of epochs for training.

- `--batch_size` **BATCH_SIZE**  
  Batch size for Training.

- `--patience` **PATIENCE**  
  Patience for Early Stopping to terminate training.

- `--train_workers` **TRAIN_WORKERS**  
  Number of Workers for training Data Loaders.

- `--val_workers` **VAL_WORKERS**  
  Number of Workers for validation Data Loaders.

- `--random_state` **RANDOM_STATE**  
  Random state for the Train-Test Data Split.

- `--batch_sampling` **{mixed,sequential}**  
  Type of Batch-Sampling for Training, either `sequential` or `mixed`.

- `--method` **{AbsWeighting,EW,GradNorm,MGDA,UW,DWA,GLS,GradDrop,PCGrad,GradVac,IMTL,CAGrad,Nash_MTL,RLW,MoCo,Aligned_MTL,DB_MTL,STCH,ExcessMTL,FairGrad}**  
  Type of Loss Weighting Method.

- `--train_rule` **{None,CBReweight,IBReweight,DRW}**  
  Type of Loss Rules Method.

- `--loss_type` **{CE,LDAM,Focal,IB,IBFocal,F-CB,CE-CB,LA,CDT,VS}**  
  Type of Loss Function.

- `--beta` **BETA**  
  Beta value for CBL Loss and DRW.

- `--pretrained_lr` **PRETRAINED_LR**  
  Learning Rate for the unfrozen pre-trained layers of the Backbone.

- `--pretrained_weight_decay` **PRETRAINED_WEIGHT_DECAY**  
  Weight Decay for the unfrozen pre-trained layers of the Backbone.

- `--scratch_lr` **SCRATCH_LR**  
  Learning Rate for the FC layers to be trained from scratch.

- `--scratch_weight_decay` **SCRATCH_WEIGHT_DECAY**  
  Weight Decay for the FC layers to be trained from scratch.

# Inference Arguments
- `--img_dir` **IMG_DIR**  
  Directory path that contains the image files for Training.

- `--csv_path` **CSV_PATH**  
  CSV file path that contains the ground truth for Training.

- `--ckpt_path` **CKPT_PATH**  
  Checkpoint file path to be used for inference.

- `--attributes_config` **ATTRIBUTES_CONFIG**  
  JSON file path that contains the attributes and their list of labels for each category. Defaults to `config/attrs_config.json`

- `--out_csv_path` **OUT_CSV_PATH**  
  Output file path with the filename to store the inference as a CSV. By default, it is saved in the same folder as the checkpoints.

