#!/bin/bash

IMG_DIR=""
CSV_PATH=""
OUTPUT_DIR="./output"

# If need WandB, then fill the below variables and delete --no-wandb argument.
WANDB_API_KEY=""
WANDB_RUN_NAME="VT-Train"
WANDB_PROJECT_NAME="VT-Challenge"

 # If there is any issue with reproducing results, Use train/val split used to create the model
 # by uncommenting L197 and L198 in train/trainer.py.

WANDB_API_KEY=${WANDB_API_KEY} python train_model.py --img_dir ${IMG_DIR} \
                                                      --csv_path ${CSV_PATH} \
                                                      --inter_linear \
                                                      --batch_norm \
                                                      --unfreeze_proj \
                                                      --train_rule DRW \
                                                      --loss_type LDAM \
                                                      --no-wandb \
                                                      --run_name ${WANDB_RUN_NAME} \
                                                      --wandb_project ${WANDB_PROJECT_NAME} \
                                                      --patience 30 \
                                                      --save_dir ${OUTPUT_DIR} \
