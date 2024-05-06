from decord import VideoReader
import cv2
import numpy as np
import json 
import os
from tqdm import tqdm
from multiprocessing import Pool
import copy
import argparse

phase_number_map = {
    '0': 'prerecognition',
    '1': 'recognition',
    '2': 'judgement',
    '3': 'action',
    '4': 'avoidance'
}


def extract_frames(video_path, frame_indices, original_frame_indices):
    vr = VideoReader(video_path)
    if frame_indices[-1] == len(vr):
        frame_indices[-1] = len(vr) - 1
    frames = {ori_idx: vr[frame_idx].asnumpy() for frame_idx, ori_idx in zip(frame_indices, original_frame_indices)}
    return frames


def draw_and_save_bboxes(key, frames, ped_bboxes, veh_bboxes, phase_numbers, phase_number_map):
    for frame_id, frame_np in frames.items():
        frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        if str(frame_id) in ped_bboxes:
            bbox = ped_bboxes[str(frame_id)]
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=4)
        if str(frame_id) in veh_bboxes:
            bbox = veh_bboxes[str(frame_id)]
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255, 0, 0), thickness=4)
 
        phase_number = phase_numbers.get(str(frame_id), "")
        if str(phase_number):
            if 'BDD' in key:
                file_name = key.replace('.mp4', f'_{phase_number_map[str(phase_number)]}.jpg').replace('/videos', '/bbox_global')
                dirname = os.path.dirname(file_name)
                os.makedirs(dirname, exist_ok=True)
            else:
                key = key.replace('.mp4', '/').replace('/videos', '/bbox_global')
                os.makedirs(key, exist_ok=True)
                file_name = f"{key}{phase_number}_{phase_number_map[str(phase_number)]}.jpg"
            
            cv2.imwrite(file_name, frame)


def enlarge_bbox(bbox, scale=1.2):
    xmin, ymin, width, height = bbox
    center_x, center_y = xmin + width / 2, ymin + height / 2
    
    new_width = width * scale
    new_height = height * scale

    new_xmin = center_x - new_width / 2
    new_ymin = center_y - new_height / 2

    return new_xmin, new_ymin, new_width, new_height


def enlarge_bbox_square(bbox, scale=1.2):
    xmin, ymin, width, height = bbox
    center_x, center_y = xmin + width / 2, ymin + height / 2
    
    new_width = width * scale
    new_height = height * scale

    new_height, new_width = max(new_width, new_height), max(new_width, new_height) # Not used when draw bbox
    
    new_xmin = center_x - new_width / 2
    new_ymin = center_y - new_height / 2
    
    return new_xmin, new_ymin, new_width, new_height


def calculate_combined_bbox(bbox1, bbox2):
    xmin = min(bbox1[0], bbox2[0])
    ymin = min(bbox1[1], bbox2[1])
    xmax = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    ymax = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    
    return xmin, ymin, xmax - xmin, ymax - ymin


def constrain_bbox_within_frame(bbox, frame_shape):
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(frame_shape[1], int(xmax))
    ymax = min(frame_shape[0], int(ymax))
    return xmin, ymin, xmax, ymax


def draw_and_save_bboxes_scale_version(key, frames, ped_bboxes, veh_bboxes, phase_numbers, phase_number_map, scale=1.5):
    for frame_id, frame_np in frames.items():
        frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        combined_bbox = None

        # Enlarge and draw pedestrian bbox
        if str(frame_id) in ped_bboxes:
            bbox = enlarge_bbox(ped_bboxes[str(frame_id)])
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height
            xmin, ymin, xmax, ymax = constrain_bbox_within_frame((xmin, ymin, xmax, ymax), frame.shape)
            combined_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=3)

        # Enlarge and draw vehicle bbox
        if str(frame_id) in veh_bboxes:
            bbox = enlarge_bbox(veh_bboxes[str(frame_id)])
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height
            xmin, ymin, xmax, ymax = constrain_bbox_within_frame((xmin, ymin, xmax, ymax), frame.shape)
            if combined_bbox is not None:
                combined_bbox = calculate_combined_bbox(combined_bbox, (xmin, ymin, xmax - xmin, ymax - ymin))
            else:
                combined_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255, 0, 0), thickness=3)

        # Enlarge the combined bbox
        if combined_bbox is not None:
            min_area = 0.1
            max_area = 0.6
            area_ratio = (combined_bbox[-2] * combined_bbox[-1]) / (frame.shape[0] * frame.shape[1])
            try:
                if combined_bbox[-2] / combined_bbox[-1] > 4 or combined_bbox[-1] / combined_bbox[-2] > 4:
                    width_ratio, height_ratio = combined_bbox[-2] / frame.shape[1], combined_bbox[-1] / frame.shape[0]
                    area_ratio = max(width_ratio, height_ratio)
            except:
                print(f"[WARRNING]: Zero detected: {combined_bbox}")

            min_scale = 1.0
            max_scale = 3.0

            ratio = min(max_area, max(min_area, area_ratio))
            # print(ratio)
            # scale = -4 * ratio + 3.4 
            
            combined_bbox = enlarge_bbox_square(combined_bbox, scale=scale)
            xmin, ymin, width, height = combined_bbox
            xmax, ymax = int(xmin + width), int(ymin + height)
            xmin, ymin = int(xmin), int(ymin)
            xmin, ymin, xmax, ymax = constrain_bbox_within_frame((xmin, ymin, xmax, ymax), frame.shape)
            cropped_frame = frame[ymin:ymax, xmin:xmax]
        else:
            cropped_frame = frame

        
        # Get the corresponding phase number
        if str(frame_id) in phase_numbers:
            phase_number = phase_numbers[str(frame_id)]
        else:
            phase_number = ''

        if str(phase_number):
            if 'BDD' in key:
                file_name = key.replace('.mp4', f'_{phase_number_map[str(phase_number)]}.jpg').replace('/videos', '/bbox_local')
                dirname = os.path.dirname(file_name)
                os.makedirs(dirname, exist_ok=True)
            else:
                key = key.replace('.mp4', '/').replace('/videos', '/bbox_local')
                os.makedirs(key, exist_ok=True)
                file_name = f"{key}{phase_number}_{phase_number_map[str(phase_number)]}.jpg"
            
            if cropped_frame.size > 0:
                cv2.imwrite(file_name, cropped_frame)
            else:
                print(cropped_frame.shape)
                print(f"Empty cropped frame for frame ID {key} {frame_id} {ped_bboxes[str(frame_id)]} {combined_bbox}. Skipping save.")


def process_video(args):
    video_path, data, phase_number_map, scale = args
    frame_indices = list(map(int, data["phase_number"].keys()))
    if len(frame_indices) == 0:
        return
    frame_indices_process = copy.deepcopy(frame_indices)
    if 'fps' in data:
        if float(data['fps']) > 40.0:
            for i in range(len(frame_indices)):
                frame_indices_process[i] = frame_indices_process[i] // 2
    frames = extract_frames(video_path, frame_indices_process, frame_indices)
    draw_and_save_bboxes(
        video_path,
        frames, 
        data["ped_bboxes"], 
        data["veh_bboxes"], 
        data["phase_number"], 
        phase_number_map, 
    )
    draw_and_save_bboxes_scale_version(
        video_path,
        frames, 
        data["ped_bboxes"], 
        data["veh_bboxes"], 
        data["phase_number"], 
        phase_number_map,
        scale
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno', type=str, help='File with bbox anno')
    parser.add_argument('--worker', type=int, default=1, help='process num (CPU count)')
    parser.add_argument('--scale', type=float, default=1.5, help='scale up coefficient')
    args = parser.parse_args()
    anno = json.load(open(args.anno))
    num_processes = args.worker 
    with Pool(processes=num_processes) as pool:
        jobs = []
        for video_path, data in tqdm(anno.items()):
            job = (video_path, data, phase_number_map, args.scale)
            jobs.append(job)
        results = list(tqdm(pool.imap(process_video, jobs), total=len(jobs)))