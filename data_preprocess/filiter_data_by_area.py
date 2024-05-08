import os, json, csv
from tqdm import tqdm

part = 'train'
root_path = './data'
reference_view_path = './data/test_part/view_used_as_main_reference_for_multiview_scenario.csv'

data_path = './processed_anno/llava_format/wts_bdd_llava_qa_train_stage.json'
save_path = './processed_anno/llava_format/wts_bdd_llava_qa_train_stage_filted.json'

area_thr = 1000
stage_map = {'prerecognition': 0, 'recognition': 1, 'judgement': 2, 'action': 3, 'avoidance': 4}

# good_view for overhead
with open(reference_view_path, 'r') as file:
    reference_views = {}
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        reference_views[row[0]] = row[1:]


with open(data_path, 'r') as f:
    data_json = json.load(f)

new_data = []
for data in tqdm(data_json):
    view = data['image'].split('/')[-2] if 'WTS' in data['image'] else data['image'].split('/')[-1].split('_')[0]
    if 'WTS' in data['image'] and 'overhead_view' in data['image']:     
        if data['id'] in reference_views.keys():               #  监控视角过滤
            if view + '.mp4' not in reference_views[data['id']]:    
                print(f"view filter: {data['image']}, {view}")
                continue 

    satge = data['image'].split('/')[-1].split('.')[0].split('_')[-1]

    pedestrian_box_path, vehicle_box_path = '', ''
    pedestrian_bbox, vehicle_bbox = '', ''
    if 'WTS' in data['image']: 
        if 'normal_trimmed' not in data['image']:                         
            if 'overhead_view' in data['image']:
                pedestrian_box_path = f"{root_path}/WTS/annotations/bbox_annotated/pedestrian/{part}/{data['id']}/overhead_view/{view}_bbox.json"   
                vehicle_box_path = f"{root_path}/WTS/annotations/bbox_annotated/vehicle/{part}/{data['id']}/overhead_view/{view}_bbox.json"         
            elif 'vehicle_view' in data['image']:
                pedestrian_box_path = f"{root_path}/WTS/annotations/bbox_annotated/pedestrian/{part}/{data['id']}/vehicle_view/{view}_bbox.json"   
                vehicle_box_path = f"{root_path}/WTS/annotations/bbox_annotated/vehicle/{part}/{data['id']}/vehicle_view/{view}_bbox.json"    
        else:
            if 'overhead_view' in data['image']:
                pedestrian_box_path = f"{root_path}/WTS/annotations/bbox_annotated/pedestrian/{part}/normal_trimmed/{data['id']}/overhead_view/{view}_bbox.json"   
                vehicle_box_path = f"{root_path}/WTS/annotations/bbox_annotated/vehicle/{part}/normal_trimmed/{data['id']}/overhead_view/{view}_bbox.json"         
            elif 'vehicle_view' in data['image']:
                pedestrian_box_path = f"{root_path}/WTS/annotations/bbox_annotated/pedestrian/{part}/normal_trimmed/{data['id']}/vehicle_view/{view}_bbox.json"   
                vehicle_box_path = f"{root_path}/WTS/annotations/bbox_annotated/vehicle/{part}/normal_trimmed/{data['id']}/vehicle_view/{view}_bbox.json"   

    elif 'BDD_PC_5k' in data['image']: 
        pedestrian_box_path = f"{root_path}/BDD_PC_5k/annotations/bbox_annotated/{part}/{view}_bbox.json"      
    
    try:
        pedestrian_bbox = json.load(open(pedestrian_box_path))
    except:
        print(f"no pedestrian json filter: {data['image']}")
        # miss_data.append(f"no pedestrian json filter: {data['image']}")
    try:
        vehicle_bbox = json.load(open(vehicle_box_path))
    except:
        pass
        # print(f"no vehicle json filter: {data['image']}")
        # miss_data.append(f"no vehicle json filter: {data['image']}")

    if pedestrian_bbox != '':
        pedestrian_box = ''
        for single_box in pedestrian_bbox['annotations']:
            if str(stage_map[satge]) == single_box['phase_number'] or stage_map[satge] == single_box['phase_number']:
                pedestrian_box = single_box
        if pedestrian_box != '':
            human_area = pedestrian_box['bbox'][2]*pedestrian_box['bbox'][3]    
        else:
            human_area = 0
    else:
        human_area = 0
        # print(f"no human box filter: {data['image']}")
        # miss_data.append(f"no human box filter: {data['image']}")


    if vehicle_bbox != '':
        vehicle_box=''
        for single_box in vehicle_bbox['annotations']:
            if str(stage_map[satge]) == single_box['phase_number'] or stage_map[satge] == single_box['phase_number']:
                vehicle_box = single_box
        if vehicle_box != '':
            vehicle_area = vehicle_box['bbox'][2]*vehicle_box['bbox'][3] 
        else:
            vehicle_area = 0    
    else:
        vehicle_area = 0
        # print(f"no vehicle box filter: {data['image']}")
        # miss_data.append(f"no vehicle box filter: {data['image']}")
    

    if human_area > area_thr and vehicle_area > area_thr:   # 人车框都正常
        new_data.append(data)
    elif human_area > area_thr and vehicle_area == 0:   # 只有人，人大于阈值，没有车
        new_data.append(data)
    elif vehicle_area > area_thr and human_area == 0:   # 只有车，车大于阈值，没有人
        new_data.append(data)
    else:
        print(f"area filter:{data['image']}, {human_area}")
   
    
print(f'num:{len(data_json)} vs {len(new_data)}')

with open(save_path, 'w') as f:
    f.write(json.dumps(new_data, indent=2, ensure_ascii=False))
