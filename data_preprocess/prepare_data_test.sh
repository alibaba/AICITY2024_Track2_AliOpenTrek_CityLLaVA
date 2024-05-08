#!/bin/bash

num_worker=32
test_root="./data/test_part" # Store original test data
generate_test_frames_path="./data/generate_test_frames" # extract frames for evaluation
save_folder="./processed_anno" # Store json files 
scale=1.5

python extract_wts_test_frame_bbox_anno.py --root $test_root --save-folder $save_folder/frame_bbox_anno
python extract_bdd_test_frame_bbox_anno.py --root $test_root --save-folder $save_folder/frame_bbox_anno

for file in "$save_folder/frame_bbox_anno"/*test*; do
    python draw_bbox_on_frame.py --worker $num_worker --anno $file --scale $scale
done

python best_view_selection.py \
    --test-root $test_root \
    --save-path $save_folder/best_view_for_test.json \

python generate_test_frames.py \
    --root $generate_test_frames_path \
    --best-view-anno $save_folder/best_view_for_test.json \
    --bdd-test-folder $test_root/WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/bbox_global/test/public \
    --wts-test-folder $test_root/WTS_DATASET_PUBLIC_TEST/bbox_global/test/public \
    --save-folder $save_folder

echo " Testsets prepared."