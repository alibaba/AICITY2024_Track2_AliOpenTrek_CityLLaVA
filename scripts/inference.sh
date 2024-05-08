#!/bin/bash

work_dir="./llava/serve"

cd $work_dir

# batch_inference_block.py & cli_final.py

DATA_PATH="../../data_preprocess/data/generate_test_frames/bbox_global"
LOCAL_IMAGE_DATA_PATH="../../data_preprocess/data/generate_test_frames/bbox_local"
FINETUNE_MODEL="../../checkpoints/llava1_6-34b-aicity-block-single-round-bigsmall-0325" # Download the model and put it into 'checkpoints' dir.
SAVE_PATH="../../results/inference_result.json" # You can change the other directory.
NUM_POOL=1 # equal to your the number of GPU
BEST_VIEW_MAP="../../data_preprocess/processed_anno/perspective_test_images.json"

python batch_inference_block.py \
    --data-path $DATA_PATH \
    --local-image-data-path $LOCAL_IMAGE_DATA_PATH \
    --finetune-model $FINETUNE_MODEL \
    --save-path $SAVE_PATH \
    --num-pool $NUM_POOL \
    --best-view-map $BEST_VIEW_MAP