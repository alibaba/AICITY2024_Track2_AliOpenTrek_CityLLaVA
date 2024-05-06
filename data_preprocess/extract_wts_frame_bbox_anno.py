import json
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/mnt/data/AICITY2024/', help='data root path')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--save-folder', type=str, default='processed_anno', help='dirname for saving json file')

args = parser.parse_args()

video_path = os.path.join(args.root, 'WTS/videos', args.split)
annotation_path = os.path.join(args.root, 'WTS/annotations/caption', args.split)
bbox_path = os.path.join(args.root, 'WTS/annotations/bbox_annotated')

video_with_bbox_results = dict()

for item in tqdm(os.listdir(video_path)):
    if 'normal' in item:
        continue
    
    for view in ['overhead', 'vehicle']:
        current_view = os.path.join(video_path, item, f'{view}_view')

        caption_anno_path = os.path.join(annotation_path, item, f'{view}_view', f'{item}_caption.json')

        # vehicle bbox extraction
        if view == 'overhead':
            assert os.path.exists(caption_anno_path), f'{caption_anno_path} not exists'
        try:
            vehicle_annotation = json.load(open(caption_anno_path))['event_phase']
        except:
            continue
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
        
        for camera in os.listdir(current_view):
            camera_base = camera.replace('.mp4', '')
            video_with_bbox_results[os.path.join(current_view, camera)] = dict(start_time=start_time, end_time=end_time, ped_bboxes=dict(), veh_bboxes=dict(), phase_number=dict())
            pedestrian_bbox_anno_path = os.path.join(bbox_path, 'pedestrian', args.split, item, f'{view}_view', f'{camera_base}_bbox.json')
    
            if os.path.exists(pedestrian_bbox_anno_path):
                pedestrian_bbox = json.load(open(pedestrian_bbox_anno_path))['annotations']
                for bbox in pedestrian_bbox:
                    video_with_bbox_results[os.path.join(current_view, camera)]['ped_bboxes'][bbox['image_id']] = bbox['bbox']
                    video_with_bbox_results[os.path.join(current_view, camera)]['phase_number'][bbox['image_id']] = bbox['phase_number']
                
            vehicle_bbox_anno_path = os.path.join(bbox_path, 'vehicle', args.split, item, f'{view}_view', f'{camera_base}_bbox.json')
            if os.path.exists(vehicle_bbox_anno_path):
                vehicle_bbox = json.load(open(vehicle_bbox_anno_path))['annotations']
                for bbox in vehicle_bbox:
                    video_with_bbox_results[os.path.join(current_view, camera)]['veh_bboxes'][bbox['image_id']] = bbox['bbox']
                    video_with_bbox_results[os.path.join(current_view, camera)]['phase_number'][bbox['image_id']] = bbox['phase_number']


for item in tqdm(os.listdir(os.path.join(video_path, 'normal_trimmed'))):
    ori_time = item
    item = f'normal_trimmed/{item}'
    for view in ['overhead', 'vehicle']:
        current_view = os.path.join(video_path, item, f'{view}_view')

        caption_anno_path = os.path.join(annotation_path, item, f'{view}_view', f'{ori_time}_caption.json')

        # vehicle bbox extraction
        if view == 'overhead':
            assert os.path.exists(caption_anno_path), caption_anno_path
        try:
            vehicle_annotation = json.load(open(caption_anno_path))['event_phase']
        except:
            continue
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
        
        for camera in os.listdir(current_view):
            camera_base = camera.replace('.mp4', '')
            video_with_bbox_results[os.path.join(current_view, camera)] = dict(start_time=start_time, end_time=end_time, ped_bboxes=dict(), veh_bboxes=dict(), phase_number=dict())
            pedestrian_bbox_anno_path = os.path.join(bbox_path, 'pedestrian', args.split, item, f'{view}_view', f'{camera_base}_bbox.json')
    
            if os.path.exists(pedestrian_bbox_anno_path):
                pedestrian_bbox = json.load(open(pedestrian_bbox_anno_path))['annotations']
                for bbox in pedestrian_bbox:
                    video_with_bbox_results[os.path.join(current_view, camera)]['ped_bboxes'][bbox['image_id']] = bbox['bbox']
                    video_with_bbox_results[os.path.join(current_view, camera)]['phase_number'][bbox['image_id']] = bbox['phase_number']
                
            vehicle_bbox_anno_path = os.path.join(bbox_path, 'vehicle', args.split, item, f'{view}_view', f'{camera_base}_bbox.json')
            if os.path.exists(vehicle_bbox_anno_path):
                vehicle_bbox = json.load(open(vehicle_bbox_anno_path))['annotations']
                for bbox in vehicle_bbox:
                    video_with_bbox_results[os.path.join(current_view, camera)]['veh_bboxes'][bbox['image_id']] = bbox['bbox']
                    video_with_bbox_results[os.path.join(current_view, camera)]['phase_number'][bbox['image_id']] = bbox['phase_number']

os.makedirs(args.save_folder, exist_ok=True)
with open(os.path.join(args.save_folder, f'wts_{args.split}_all_video_with_bbox_anno_first_frame.json'), 'w') as f:
    f.write(json.dumps(video_with_bbox_results, indent=4))