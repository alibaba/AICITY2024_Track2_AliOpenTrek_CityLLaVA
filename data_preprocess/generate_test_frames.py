import json
import os
import cv2
import shutil
from tqdm import tqdm

number_phrase_map = {
    'prerecognition': '0',
    'recognition': '1',
    'judgement': '2',
    'action': '3',
    'avoidance': '4'
}

phrase_number_map = {v:k for k, v in number_phrase_map.items()}

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/mnt/data/AICITY2024/image_for_test', help='data root path')
parser.add_argument('--best-view-anno', type=str, required=True, help='best view anno for selecting frames for test')
parser.add_argument('--wts-test-folder', type=str, required=True)
parser.add_argument('--bdd-test-folder', type=str, required=True)
parser.add_argument('--save-folder', type=str, default='./processed_anno')

args = parser.parse_args()

best_view = json.load(open(args.best_view_anno))
root = os.path.join(args.root, 'bbox_global')

camera_path_mapping = dict()

global_image_path = os.path.join(args.wts_test_folder)
for event in os.listdir(global_image_path):
    if 'normal_trimmed' in event:
        continue
    for view in os.listdir(os.path.join(global_image_path, event)):
        parent_path = os.path.join(global_image_path, event, view)
        for camera in os.listdir(parent_path):
            camera_path_mapping[camera] = os.path.join(parent_path, camera)
    

for event in os.listdir(os.path.join(global_image_path, 'normal_trimmed')):
    for view in os.listdir(os.path.join(global_image_path, 'normal_trimmed', event)):
        parent_path = os.path.join(global_image_path, 'normal_trimmed', event, view)
        for camera in os.listdir(parent_path):
            camera_path_mapping[camera] = os.path.join(parent_path, camera)

perspective = dict()
for key, value in tqdm(best_view.items()):
    os.makedirs(os.path.join(root, key), exist_ok=True)
    os.makedirs(os.path.join(root.replace('bbox_global', 'bbox_local'), key), exist_ok=True)
    for label, segment in phrase_number_map.items():
        if 'video' in key:
            image_name = f'{key}_{segment}.jpg'
            try:                
                shutil.copy(os.path.join(args.bdd_test_folder, image_name), os.path.join(root, key, f'{label}.jpg'))
                shutil.copy(os.path.join(args.bdd_test_folder.replace('bbox_global', 'bbox_local'), image_name), os.path.join(root.replace('bbox_global', 'bbox_local'), key, f'{label}.jpg'))
            except:
                print(f'{key}_{segment} not exist!')
        else:
            image_name = f'{label}_{segment}.jpg'
            try:                
                shutil.copy(os.path.join(camera_path_mapping[value.replace('.mp4', '')], image_name), os.path.join(root, key, f'{label}.jpg'))
                shutil.copy(os.path.join(camera_path_mapping[value.replace('.mp4', '')], image_name).replace('bbox_global', 'bbox_local'), os.path.join(root.replace('bbox_global', 'bbox_local'), key, f'{label}.jpg'))
            except:
                print(f'{value.replace(".mp4", "")}/{image_name} not exist!')
        
        if os.path.exists(os.path.join(root, key, f'{label}.jpg')):
            perspective[os.path.abspath(os.path.join(root, key, f'{label}.jpg'))] = 'vehicle' if 'vehicle' in key or 'video' in key else 'overhead'


def find_closest_number(target, arr):
    min_diff = float('inf')
    closest_num = None
    for num in arr:
        diff = abs(target - int(num))
        
        if diff < min_diff:
            min_diff = diff
            closest_num = num
            
    return closest_num

no_bbox_frames_ones = set()
for item in os.listdir(root):
    images = os.listdir(os.path.join(root, item))
    if len(images) == 0:
        no_bbox_frames_ones.add(item)
        continue
    if len(images) != 5:
        for image in ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg']:
            if image not in images:
                closest_image = find_closest_number(int(image.split('.')[0]), [int(x.split('.')[0]) for x in images])
                source_path = os.path.join(root, item, f'{closest_image}.jpg')
                target_path = os.path.join(root, item, image)
                shutil.copy(source_path, target_path)

                perspective[os.path.abspath(target_path)] = perspective[os.path.abspath(source_path)]

                source_path = os.path.join(root.replace('bbox_global', 'bbox_local'), item, f'{closest_image}.jpg')
                target_path = os.path.join(root.replace('bbox_global', 'bbox_local'), item, image)
                shutil.copy(source_path, target_path)

    assert len(os.listdir(os.path.join(root, item))) == 5


def extract_frames(source_video, source_anno, save_folder):
    with open(source_anno, 'r') as f:
        data = json.load(f)
    
    cap = cv2.VideoCapture(source_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for event in data['event_phase']:
        label = event['labels'][0]
        start_time = float(event['start_time'])

        frame_number = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f'{save_folder}/{label}.jpg', frame)
            cv2.imwrite(f'{save_folder.replace("bbox_global", "bbox_local")}/{label}.jpg', frame)
        else:
            print(f"Error: Unable to extract frame for event {label}, source video: {source_video}")

    cap.release()

for item in no_bbox_frames_ones:
    if 'video' in item:
        source_video = f'{args.bdd_test_folder.replace("bbox_global", "videos")}/{item}.mp4'
        source_anno = f'{args.bdd_test_folder.replace("bbox_global", "annotations/caption")}_challenge/{item}_caption.json'
        view = 'vehicle'
    else:
        best_view_video = best_view[item]
        if 'normal' in best_view_video:
            if 'vehicle' in best_view_video:
                source_video = f'{args.wts_test_folder.replace("bbox_global", "videos")}/normal_trimmed/{item}/vehicle_view/{best_view_video}'
                source_anno = f'{args.wts_test_folder.replace("bbox_global", "annotations/caption")}_challenge/normal_trimmed/{item}/vehicle_view/{item}_caption.json'
                view = 'vehicle'
            else:
                source_video = f'{args.wts_test_folder.replace("bbox_global", "videos")}/normal_trimmed/{item}/overhead_view/{best_view_video}'
                source_anno = f'{args.wts_test_folder.replace("bbox_global", "annotations/caption")}_challenge/normal_trimmed/{item}/overhead_view/{item}_caption.json'
                view = 'overhead'
        else:
            if 'vehicle' in best_view_video:
                source_video = f'{args.wts_test_folder.replace("bbox_global", "videos")}/{item}/vehicle_view/{best_view_video}'
                source_anno = f'{args.wts_test_folder.replace("bbox_global", "annotations/caption")}_challenge/{item}/vehicle_view/{item}_caption.json'
                view = 'vehicle'
            else:
                source_video = f'{args.wts_test_folder.replace("bbox_global", "videos")}/{item}/overhead_view/{best_view_video}'
                source_anno = f'{args.wts_test_folder.replace("bbox_global", "annotations/caption")}_challenge/{item}/overhead_view/{item}_caption.json'
                view = 'overhead'

    save_folder = os.path.join(root, item)
    extract_frames(source_video, source_anno, save_folder)
    for image in ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg']:
        target_path = os.path.abspath(os.path.join(root, item, image))
        assert os.path.exists(target_path)
        perspective[target_path] = view

for item in os.listdir(root):
    images = os.listdir(os.path.join(root, item))
    if len(images) != 5:
        print(item)

with open(os.path.join(args.save_folder, 'perspective_test_images.json'), 'w') as f:
    f.write(json.dumps(perspective, indent=4))