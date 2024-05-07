
import os
import argparse
import torch
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Receive deepen model's args")
    parser.add_argument("--model_path", default='./models/llava-v1.6-34b', type=str, help="original model path")
    parser.add_argument("--output_path", default='./models/llava-v1.6-34b-12block', type=str, help="deepened model ckpt save path")
    parser.add_argument("--original_layers", default=60, type=int, help="original model num layers")
    parser.add_argument("--layers", default=72, type=int, help="deepen model num layers")

    # Parse the arguments
    args = parser.parse_args()
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, _ = load_pretrained_model(args.model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    ckpt = model.state_dict()
    
    split = int(args.original_layers / (args.layers - args.original_layers))
    layer_cnt = 0

    output = {}
    for i in tqdm(range(args.original_layers)):
        for k in ckpt:
            if ('layers.' + str(i) + '.') in k:
                output[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = ckpt[k]
        layer_cnt += 1
        if (i+1) % split == 0:
            for k in ckpt:
                if ('layers.' + str(i) + '.') in k:
                    if 'down_proj' in k or 'o_proj' in k:
                        output[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = torch.zeros_like(ckpt[k])
                    else:
                        output[k.replace(('layers.' + str(i) + '.'), ('layers.' + str(layer_cnt) + '.'))] = ckpt[k]


            layer_cnt += 1
        
    assert layer_cnt==args.layers
    add_layer = [(split+1)*i+split for i in range(0, args.layers-args.original_layers)]
    print(add_layer)
    for k in ckpt:
        if not 'layers' in k:
            output[k] = ckpt[k]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    torch.save(output, args.output_path + '/pytorch_model.bin')

if __name__ == "__main__":
    main()