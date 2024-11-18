#!/bin/bash

TEST_IMG_DIR=""
TEST_CSV_PATH=""
CKPT_PATH=""

# Output CSV will be stored in the directory Checkpoint itself in the name of the checkpoint

python inference_model.py --ckpt_path ${CKPT_PATH} \
                          --img_dir ${TEST_IMG_DIR} \
                          --csv_path ${TEST_CSV_PATH}
