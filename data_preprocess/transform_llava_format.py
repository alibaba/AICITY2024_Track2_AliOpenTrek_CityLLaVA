import os
import json
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/mnt/data/AICITY2024/', help='data root path')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--save-folder', type=str, default='processed_anno', help='dirname for saving json file')
parser.add_argument('--wts-global-image-path', type=str, required=True, help='root path for wts global images')
parser.add_argument('--bdd-global-image-path', type=str, required=True, help='root path for bdd global images')

args = parser.parse_args()

root = args.root

phrase_number_map = {
    '0': 'prerecognition',
    '1': 'recognition',
    '2': 'judgement',
    '3': 'action',
    '4': 'avoidance'
}
number_phrase_map = {v: k for k, v in phrase_number_map.items()}

camera_path_mapping = dict()

wts_anno_path = os.path.join(root, 'WTS/annotations/caption', args.split)
bdd_anno_path = os.path.join(root, 'BDD_PC_5k/annotations/caption', args.split)

train_samples = list()

overhead = 'overhead_view'
vehicle = 'vehicle_view'

for item in os.listdir(wts_anno_path):
    overhead_flag, vehicle_flag = True, True
    try:
        overhead_view = json.load(open(f'{wts_anno_path}/{item}/{overhead}/{item}_caption.json'))
    except:
        overhead_flag = False
    try:
        vehicle_view = json.load(open(f'{wts_anno_path}/{item}/{vehicle}/{item}_caption.json'))
    except:
        vehicle_flag = False
    sample_id = item
    
    if overhead_flag:
        for event in overhead_view['event_phase']:
            cur_data = dict()
            cur_data['id'] = sample_id
            cur_data['segment'] =  phrase_number_map[event['labels'][0]]
            cur_data['view'] = 'overhead'
            cur_data['start_time'] = event['start_time']
            cur_data['end_time'] = event['end_time']
            cur_data['conversations'] = list()

            cur_data['conversations'].append({
                'from': 'human',
                'value': '<image>\nPlease describe the interested pedestrian in the video.'
            })

            cur_data['conversations'].append({
                'from': 'gpt',
                'value': event['caption_pedestrian']
            })

            cur_data['conversations'].append({
                'from': 'human',
                'value': 'Please describe the interested vehicle in the video.'
            })

            cur_data['conversations'].append({
                'from': 'gpt',
                'value': event['caption_vehicle']
            })

            for image in overhead_view['overhead_videos']:
                cur_data['image'] =  image
                train_samples.append(copy.deepcopy(cur_data))

    if vehicle_flag:
        for event in vehicle_view['event_phase']:
            cur_data = dict()
            cur_data['id'] = sample_id
            cur_data['segment'] =  phrase_number_map[event['labels'][0]]
            cur_data['view'] = 'vehicle'
            cur_data['start_time'] = event['start_time']
            cur_data['end_time'] = event['end_time']
            cur_data['conversations'] = list()

            cur_data['conversations'].append({
                'from': 'human',
                'value': '<image>\nPlease describe the interested pedestrian in the video.'
            })

            cur_data['conversations'].append({
                'from': 'gpt',
                'value': event['caption_pedestrian']
            })

            cur_data['conversations'].append({
                'from': 'human',
                'value': 'Please describe the interested vehicle in the video.'
            })

            cur_data['conversations'].append({
                'from': 'gpt',
                'value': event['caption_vehicle']
            })

            cur_data['image'] = vehicle_view['vehicle_view']
            train_samples.append(cur_data)

for item in os.listdir(f'{wts_anno_path}/normal_trimmed'):
    overhead_flag, vehicle_flag = True, True
    try:
        overhead_view = json.load(open(f'{wts_anno_path}/normal_trimmed/{item}/{overhead}/{item}_caption.json'))
    except:
        overhead_flag = False
    try:
        vehicle_view = json.load(open(f'{wts_anno_path}/normal_trimmed/{item}/{vehicle}/{item}_caption.json'))
    except:
        vehicle_flag = False
    sample_id = item
    
    if overhead_flag:
        for event in overhead_view['event_phase']:
            cur_data = dict()
            cur_data['id'] = sample_id
            cur_data['segment'] =  phrase_number_map[event['labels'][0]]
            cur_data['view'] = 'overhead'
            cur_data['start_time'] = event['start_time']
            cur_data['end_time'] = event['end_time']
            cur_data['conversations'] = list()

            cur_data['conversations'].append({
                'from': 'human',
                'value': '<image>\nPlease describe the interested pedestrian in the video.'
            })

            cur_data['conversations'].append({
                'from': 'gpt',
                'value': event['caption_pedestrian']
            })

            cur_data['conversations'].append({
                'from': 'human',
                'value': 'Please describe the interested vehicle in the video.'
            })

            cur_data['conversations'].append({
                'from': 'gpt',
                'value': event['caption_vehicle']
            })

            for image in overhead_view['overhead_videos']:
                cur_data['image'] =  image
                train_samples.append(copy.deepcopy(cur_data))

    if vehicle_flag:
        for event in vehicle_view['event_phase']:
            cur_data = dict()
            cur_data['id'] = sample_id
            cur_data['segment'] =  phrase_number_map[event['labels'][0]]
            cur_data['view'] = 'vehicle'
            cur_data['start_time'] = event['start_time']
            cur_data['end_time'] = event['end_time']
            cur_data['conversations'] = list()

            cur_data['conversations'].append({
                'from': 'human',
                'value': '<image>\nPlease describe the interested pedestrian in the video.'
            })

            cur_data['conversations'].append({
                'from': 'gpt',
                'value': event['caption_pedestrian']
            })

            cur_data['conversations'].append({
                'from': 'human',
                'value': 'Please describe the interested vehicle in the video.'
            })

            cur_data['conversations'].append({
                'from': 'gpt',
                'value': event['caption_vehicle']
            })

            cur_data['image'] = vehicle_view['vehicle_view']
            train_samples.append(cur_data)

for item in os.listdir(bdd_anno_path):
    captions = json.load(open(f'{bdd_anno_path}/{item}'))
    sample_id = captions['id']
    for event in captions['event_phase']:
        cur_data = dict()
        cur_data['id'] = sample_id
        cur_data['segment'] =  event['labels'][0]
        cur_data['start_time'] = event['start_time']
        cur_data['end_time'] = event['end_time']
        cur_data['conversations'] = list()

        cur_data['conversations'].append({
            'from': 'human',
            'value': '<image>\nPlease describe the interested pedestrian in the video.'
        })

        cur_data['conversations'].append({
            'from': 'gpt',
            'value': event['caption_pedestrian']
        })

        cur_data['conversations'].append({
            'from': 'human',
            'value': 'Please describe the interested vehicle in the video.'
        })

        cur_data['conversations'].append({
            'from': 'gpt',
            'value': event['caption_vehicle']
        })

        cur_data['image'] = captions['video_name']
        camera_path_mapping[cur_data['image'].replace('.mp4', '')] = os.path.join(args.bdd_global_image_path, args.split)
        train_samples.append(cur_data)


global_image_path = os.path.join(args.wts_global_image_path, args.split)
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


reserved_train_samples = list()
for item in train_samples:
    image = item['image'].replace('.mp4', '')
    segment = item['segment']

    if 'video' in image:
        train_image_name = f'{image}_{segment}.jpg'
    else:
        train_image_name = f'{number_phrase_map[segment]}_{segment}.jpg'

    if image in camera_path_mapping:
        item['image'] = os.path.join(camera_path_mapping[image], train_image_name)
        if os.path.exists(item['image']):
            item['image'] = item['image'].replace('./data/', '')
            reserved_train_samples.append(item)


os.makedirs(args.save_folder, exist_ok=True)
with open(os.path.join(args.save_folder, f'wts_bdd_{args.split}.json'), 'w+') as f:
    f.write(json.dumps(reserved_train_samples, indent=4))