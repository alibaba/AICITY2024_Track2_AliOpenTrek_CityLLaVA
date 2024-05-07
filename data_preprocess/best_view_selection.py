import os, json, csv, glob
from tqdm import tqdm 
from collections import defaultdict
import argparse

def get_best_view_wts(ann_path, bbox_path, scnearios, reference_views):
    best_view_video = {}
    for scneario in tqdm(scnearios):
        if '.DS_Store' in scneario: 
            continue
        if '_normal_' in scneario:
            if os.path.exists(os.path.join(bbox_path, f'normal_trimmed/{scneario}/overhead_view')) or not os.path.exists(os.path.join(bbox_path, f'normal_trimmed/{scneario}/vehicle_view')):
                best_view_video[scneario] = scneario + '.mp4'
            else:
                best_view_video[scneario] = scneario +'_vehicle_view.mp4'
        else:
            if scneario == '20231006_18_CN29_T1':
                print('f')
            try:
                overhead_view_json = json.load(open(glob.glob(os.path.join(ann_path, f'{scneario}/overhead_view/*.json'))[0]))
            except:
                overhead_view_json = None

            views = []
            for overhand in overhead_view_json['overhead_videos']:
                if scneario in reference_views.keys():
                    if overhand in reference_views[scneario]:
                        views.append(overhand)
                else:
                    print(f'no reference view: {scneario}')
                    views.append(overhand)
            best_view_score = 0
            best_view = None
            for view in views:
                if os.path.exists(os.path.join(bbox_path, f"{scneario}/overhead_view/{view.replace('.mp4', '')}_bbox.json")):
                     bbox = json.load(open(os.path.join(bbox_path, f"{scneario}/overhead_view/{view.replace('.mp4', '')}_bbox.json")))
                elif os.path.exists(os.path.join(bbox_path, f"{scneario}/vehicle_view/{scneario}_vehicle_view_bbox.json")):
                    bbox = json.load(open(os.path.join(bbox_path, f"{scneario}/vehicle_view/{scneario}_vehicle_view_bbox.json")))
                else:
                    print(f'no bbox: {scneario}')
                    continue

                if len(bbox["annotations"]) == 5:
                    avg_human_area = sum([box['bbox'][2]*box['bbox'][3] for box in bbox["annotations"]])/5.
                    if avg_human_area > best_view_score:
                        best_view_score = avg_human_area
                        best_view = view
                    
                if best_view == None and os.path.exists(os.path.join(bbox_path, f"{scneario}/vehicle_view/{scneario}_vehicle_view_bbox.json")):
                    best_view = scneario +'_vehicle_view.mp4'
                else:
                    avg_human_area = sum([box['bbox'][2]*box['bbox'][3] for box in bbox["annotations"]])/len(bbox["annotations"])
                    if avg_human_area > best_view_score:
                        best_view_score = avg_human_area
                        best_view = view    

            # We found that the bounding boxes of 20230728_13_CN21_T1_Camera2_5.mp4 and 20230728_13_CN21_T2_Camera2_5 is incorrect
            if scneario == '20230728_13_CN21_T1' or scneario == '20230728_13_CN21_T2':
                best_view=scneario +'_vehicle_view.mp4'

            best_view_video[scneario] = best_view

    return best_view_video



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-root', type=str, default='./data/test_part')
    parser.add_argument('--save-path', type=str, default='./processed_anno/best_view_for_test.json')
    args = parser.parse_args()
    wts_ann_path = os.path.join(args.test_root, 'WTS_DATASET_PUBLIC_TEST/annotations/caption/test/public_challenge')
    wts_bbox_path = os.path.join(args.test_root, 'WTS_DATASET_PUBLIC_TEST_BBOX/annotations/bbox_annotated/pedestrian/test/public')
    bdd_video_path = os.path.join(args.test_root, 'WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/videos/test/public')
    reference_view_path = os.path.join(args.test_root, 'view_used_as_main_reference_for_multiview_scenario.csv')
    save_path = args.save_path

    # get the official recommended perspectives
    with open(reference_view_path, 'r') as file:
        reference_views = {}
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            reference_views[row[0]] = row[1:]

    rest_videos = defaultdict(list)

    # get the best bdd views 
    scnearios1 = os.listdir(wts_ann_path) 
    scnearios1.remove('normal_trimmed')
    best_view_wts1 = get_best_view_wts(wts_ann_path, wts_bbox_path, scnearios1, reference_views)
    rest_videos.update(best_view_wts1)

    scnearios2 = os.listdir(os.path.join(wts_ann_path, 'normal_trimmed'))
    best_view_wts2 = get_best_view_wts(wts_ann_path, wts_bbox_path, scnearios2, reference_views)
    rest_videos.update(best_view_wts2)

    # get the best bdd views 
    for bdd_video in os.listdir(bdd_video_path):
        rest_videos[bdd_video.split('.')[0]] = bdd_video

    with open(save_path, 'w') as f:
        f.write(json.dumps(rest_videos, indent=2, ensure_ascii=False))      
