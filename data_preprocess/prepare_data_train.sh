#!/bin/bash

num_worker=32
root="./data"
save_folder="./processed_anno" # Store json files 
splits=("train" "val")
scale=1.5

for split in "${splits[@]}"; do
    python extract_wts_frame_bbox_anno.py --root $root --save-folder $save_folder/frame_bbox_anno --split $split
    python extract_bdd_frame_bbox_anno.py --root $root --save-folder $save_folder/frame_bbox_anno --split $split
done

for file in "$save_folder/frame_bbox_anno"/*train*; do
    python draw_bbox_on_frame.py --worker $num_worker --anno $file --scale $scale
done

for file in "$save_folder/frame_bbox_anno"/*val*; do
    python draw_bbox_on_frame.py --worker $num_worker --anno $file --scale $scale
done

for split in "${splits[@]}"; do
    python transform_llava_format.py \
        --root $root \
        --save-folder $save_folder/llava_format \
        --split $split \
        --wts-global-image-path $root/WTS/bbox_global \
        --bdd-global-image-path $root/BDD_PC_5k/bbox_global
done

# generate shortQA
API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
MODEL="Qwen"

python shortQA_split.py --model $MODEL --api-key $API_KEY
python shortQA_merge.py

# data filter
python add_stage_prompt.py
python filiter_data_by_area.py
python check_image.py

echo " Trainsets prepared."