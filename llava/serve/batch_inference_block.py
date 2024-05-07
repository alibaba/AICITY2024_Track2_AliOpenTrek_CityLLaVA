import os, time, glob, json
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import argparse
import sys
import torch

from llava.serve.cli_final import infer_once

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path


def infer(model_path, model_base, data_list, i, cache_dir, best_view_map, data_path, local_image_data_path):
    # os.environ['CUDA_VISIBLE_DEVICES'] = f'{i}'

    torch.manual_seed(1234)

    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                     False, True,
                                                                     device=f"cuda:{i}")
    if model.device == torch.device('cpu'):
        print('Detecting model loaded on CPU...')
        model = model.to(f"cuda:{i}")
    results = {}
    
    for scneario in tqdm(data_list, desc=f'GPU:{str(i)}'):
        scneario_res = []
        for clip in os.listdir(os.path.join(data_path, scneario)):
            if os.path.exists(os.path.join(cache_dir, scneario, clip.replace('.jpg', '.json'))):
                res = json.load(open(os.path.join(cache_dir, scneario, clip.replace('.jpg', '.json'))))
                for key, value in res.items():
                    if key == 'labels':
                        continue
                    else:
                        value = value.replace('<|startoftext|> ', '')
                        value = value.replace('<|im_end|>', '')
            else:
                res = {}
                res['labels'] = [str(clip.replace('.jpg', ''))]
                retries = 0
                while retries < 5:
                    try:
                        global_image_path = os.path.join(data_path, scneario, clip)
                        best_view = best_view_map[global_image_path]
                        local_image_path = os.path.join(local_image_data_path, scneario, clip)
                        res_caption = infer_once(global_image_path, local_image_path, best_view, tokenizer, model, processor, context_len, model_name)
                        break
                    except Exception as e:
                        print("Error: ", e, model.device)
                        torch.cuda.empty_cache()
                        retries += 1 
                        if retries >= 5:
                            break

                res.update(res_caption)
                os.makedirs(os.path.join(cache_dir, scneario), exist_ok=True)
                with open(os.path.join(cache_dir, scneario, clip.replace('.jpg', '.json')), 'w') as f:
                    f.write(json.dumps(res, indent=4))
            
            scneario_res.append(res)
        results[scneario] = scneario_res

    return [results]


if __name__ == '__main__':
    mp.set_start_method('spawn')
    # Create the parser
    parser = argparse.ArgumentParser(description='Process the paths and configurations.')

    # Add arguments
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the global image datasets directory.')
    parser.add_argument('--local-image-data-path', type=str, required=True,
                        help='Path to the local image datasets directory.')
    parser.add_argument('--finetune-model', type=str, default=None, 
                        help='Path to the finetune model directory (if any).')
    parser.add_argument('--model-base', type=str, default=None,
                        help='Path to the model base directory (if any).')
    parser.add_argument('--save-path', type=str, required=True,
                        help='Path where the results will be saved.')
    parser.add_argument('--num-pool', type=int, default=1,
                        help='Number of pools to use, equal to the number of GPUs.')
    parser.add_argument('--cache-dir', type=str, default="test_tmp_cache", help='Store inference cache.')
    parser.add_argument('--best-view-map', type=str, required=True, help='indicate the best view')
    # Parse the arguments
    args = parser.parse_args()
    
    
    json_data = list(os.listdir(args.data_path))
    best_view_map = json.load(open(args.best_view_map))
    process_num = int(len(json_data)/args.num_pool) + 1
    pool = Pool(processes=args.num_pool)
    json_data_splits = [json_data[i:i+process_num] for i in range(0, len(json_data), process_num)]
    
    results = []

    for i, splits in enumerate(json_data_splits):
        # result = infer(finetune_model, splits[:2], i)
        result = pool.apply_async(infer, (args.finetune_model, args.model_base, splits, i, args.cache_dir, best_view_map, args.data_path, args.local_image_data_path))
        results.append(result)
    pool.close()
    pool.join()

    traffic_list = {}
    for i in range(len(results)):
        curr_dict = results[i].get()
        traffic_list.update(curr_dict[0])

    with open(args.save_path, 'w') as f:
        f.write(json.dumps(traffic_list, indent=2, ensure_ascii=False) + '\n')
