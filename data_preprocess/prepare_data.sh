#!/bin/bash

num_worker=32
root="./data"
test_root="./data/test_part" # Store original test data
generate_test_frames_path="./data/generate_test_frames"
save_folder="./processed_anno" # Store json files 
splits=("train" "val")
scale=1.5

for split in "${splits[@]}"; do
    python extract_wts_frame_bbox_anno.py --root $root --save-folder $save_folder/frame_bbox_anno --split $split
    python extract_bdd_frame_bbox_anno.py --root $root --save-folder $save_folder/frame_bbox_anno --split $split
done

python extract_wts_test_frame_bbox_anno.py --root $test_root --save-folder $save_folder/frame_bbox_anno
python extract_bdd_test_frame_bbox_anno.py --root $test_root --save-folder $save_folder/frame_bbox_anno

for file in "$save_folder/frame_bbox_anno"/*; do
    python draw_bbox_on_frame.py --worker $num_worker --anno $file --scale $scale
done

for split in "${splits[@]}"; do
    python transform_llava_format.py \
        --root $root \
        --save-folder $save_folder/llava_format \
        --split $split \
        --wts-global-image-path ./data/WTS/bbox_global \
        --bdd-global-image-path ./data/BDD_PC_5k/bbox_global
done

python best_view_selection.py \
    --test-root $test_root \
    --save-path $save_folder/best_view_for_test.json \

python generate_test_frames.py \
    --root $generate_test_frames_path \
    --best-view-anno $save_folder/best_view_for_test.json \
    --bdd-test-folder ./data/test_part/WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/bbox_global/test/public/ \
    --wts-test-folder ./data/test_part/WTS_DATASET_PUBLIC_TEST/bbox_global/test/public/ \
    --save-folder $save_folder

# generate shortQA
python shortQA_split.py
python shortQA_merge.py

# data filter
python add_stage_prompt.py
python filiter_data_by_area.py
python check_image.py

echo "Done."