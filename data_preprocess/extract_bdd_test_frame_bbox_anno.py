import json
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/mnt/data/AICITY2024/', help='data root path')
parser.add_argument('--save-folder', type=str, default='processed_anno', help='dirname for saving json file')

args = parser.parse_args()

video_root_path = os.path.join(args.root, 'WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/videos/test/public')
annotation_path = os.path.join(args.root, 'WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/annotations/caption/test/public_challenge')
bbox_path = os.path.join(args.root, 'WTS_DATASET_PUBLIC_TEST_BBOX/external/BDD_PC_5K/annotations/bbox_annotated/test/public')

video_with_bbox_results = dict()

for item in tqdm(os.listdir(video_root_path)):
    video_path = os.path.join(video_root_path, item)
    camera_base = item.replace('.mp4', '')
    overhead_caption_anno_path = os.path.join(annotation_path, f'{camera_base}_caption.json')

    # vehicle bbox extraction
    assert os.path.exists(overhead_caption_anno_path)
    vehicle_annotation = json.load(open(overhead_caption_anno_path))['event_phase']
    fps = json.load(open(overhead_caption_anno_path))['fps']
    start_time, end_time = None, None
    for phase in vehicle_annotation:
        if not start_time:
            start_time = float(phase['start_time'])
        else:
            start_time = min(float(phase['start_time']), start_time)
        if not end_time:
            end_time = float(phase['end_time'])
        else:
            end_time = max(float(phase['end_time']), end_time)

 
    video_with_bbox_results[video_path] = dict(fps=fps, start_time=start_time, end_time=end_time, ped_bboxes=dict(), veh_bboxes=dict(), phase_number=dict())
    pedestrian_bbox_anno_path = os.path.join(bbox_path, f'{camera_base}_bbox.json')

    if os.path.exists(pedestrian_bbox_anno_path):
        pedestrian_bbox = json.load(open(pedestrian_bbox_anno_path))['annotations']
        for bbox in pedestrian_bbox:
            video_with_bbox_results[video_path]['ped_bboxes'][bbox['image_id']] = bbox['bbox']
            video_with_bbox_results[video_path]['phase_number'][bbox['image_id']] = bbox['phase_number']
        
os.makedirs(args.save_folder, exist_ok=True)
with open(os.path.join(args.save_folder, 'bdd_test_all_video_with_bbox_anno_first_frame.json'), 'w') as f:
    f.write(json.dumps(video_with_bbox_results, indent=4))