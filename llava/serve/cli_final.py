import argparse
import torch
import os
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import json
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

phase_number_map = {
    '0': 'prerecognition',
    '1': 'recognition',
    '2': 'judgement',
    '3': 'action',
    '4': 'avoidance'
}

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def infer_once(input_file, local_image_path, best_view, tokenizer, model, processor, context_len, model_name, ori_conv_mode='chatml_direct'):
    # Model
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if ori_conv_mode is not None and conv_mode != ori_conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, ori_conv_mode, ori_conv_mode))
    else:
        ori_conv_mode = conv_mode
    print('conv_model: ', ori_conv_mode, model.device)
    conv = conv_templates[ori_conv_mode].copy()
    ######## notice!!!!!
    # conv.system = """<|im_start|>system
# Answer the questions. Notice that you should provide answers as detailed as possible."""
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    image = load_image(input_file)
    image_size = image.size
    # Similar operation in model_worker.py
    # local_image_path = input_file.replace('bbox_image_val_100', 'bbox_image_val_cropped_scale1_5_100')
    # local_image_path = input_file.replace('bbox_image_test_new', 'bbox_image_test_cropped_scale1_5')
    local_image = Image.open(local_image_path).convert('RGB')
    # Similar operation in model_worker.py
    image_tensor = process_images([image], processor, model.config, [local_image])
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # best_view = best_view_test[input_file.split('/')[-2]]
    # if 'BDD' in best_view:
    stage = phase_number_map[str(os.path.basename(input_file).split('.')[0])]
    if 'vehicle' in best_view:
    # if 'vehicle' in best_view_test[input_file.replace('bbox_image_val_100', 'bbox_image_val_cropped')]:
        guide_prompt = f"This is an image in '{stage}' stage. "

        questions = ['This picture shows the relationship between the pedestrian in the green bounding box and the ego-vehicle. Describe the pedestrian in the green bounding box or the pedestrian closest to the vehicle based on age, height, clothing, line of sight, relative position to the vehicle, movement status, weather conditions and road environment.', 'This picture shows the relationship between the ego-vehicle and the pedestrian in the green bounding box. Describe the ego-vehicle based on the relative position to the pedestrian, driving status, weather conditions and road environment. And describe the age, height, clothing of the pedestrian.']

        questions[1] = guide_prompt + '\n' + questions[1]
        questions[0] = guide_prompt + '\n' + questions[0]
    else:
        guide_prompt = f"This is an image in '{stage}' stage. "

        questions = ['This picture shows the relationship between the pedestrian in the green bounding box and the vehicle in the blue bounding box. Describe the pedestrian in the green bounding box or the pedestrian closest to the vehicle based on age, height, clothing, line of sight, relative position to the vehicle, movement status, weather conditions and road environment.', 'This picture shows the relationship between the vehicle in the blue bounding box and the pedestrian in the green bounding box. Describe the vehicle in the blue bounding box or the vehicle closest to the pedestrian based on the relative position to the pedestrian, driving status, weather conditions and road environment. And describe the age, height, clothing of the pedestrian.']

        questions[1] = guide_prompt + '\n' + questions[1]
        questions[0] = guide_prompt + '\n' + questions[0]


    keys = ['caption_pedestrian', 'caption_vehicle']
    keys = keys[::-1]

    questions = questions[::-1]

    # previous_prompt = [
    #                     "Describe the age, height and clothing of the pedestrian in the green box.",
    #                     "Describe the position of the pedestrian in the green box relative to the vehicle.",
    #                     "Describe the line of sight and movement status of the pedestrian in the green box.",
    #                     "Describe the weather conditions and road environment.",
    #                     "Describe the position of the vehicle in the blue box relative to the pedestrian in the green box.",
    #                     "Describe the driving status of the vehicle in the blue box."
    #                 ]

    # questions = previous_prompt + questions
    # print(questions, best_view_test[input_file.replace('bbox_image_test_new', 'bbox_image_test_cropped')])

    res = dict()
    for k in range(len(questions)):
        inp = questions[k]

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True,
                temperature=0.2,
                # num_beams=3,
                max_new_tokens=512,
                streamer=streamer,
                stopping_criteria=[stopping_criteria],
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs
        # if k >= 6:
        #     res[keys[k - 6]] = outputs.replace('<|startoftext|> ', '').replace('<|im_end|>', '')
        res[keys[k]] = outputs.replace('<|startoftext|> ', '').replace('<|im_end|>', '')

    return res


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", type=str, default="/mnt/workspace/workgroup/chengxiang/work_dirs/llava1_6-34b-aicity-0312-lora")
#     parser.add_argument("--model-base", type=str, default="/mnt/workspace/workgroup/chengxiang/models/llava-v1.6-34b")
#     parser.add_argument("--image-file", type=str, default='/mnt/workspace/workgroup/chenghao/video_analysis/dataset/BDD_PC_5k/bbox_visualization/test/video57_4.jpg')
#     parser.add_argument("--device", type=str, default="cuda")
#     parser.add_argument("--conv-mode", type=str, default="chatml_direct")
#     parser.add_argument("--temperature", type=float, default=0.2)
#     parser.add_argument("--max-new-tokens", type=int, default=512)
#     parser.add_argument("--load-8bit", default=False)
#     parser.add_argument("--load-4bit", default=False)
#     parser.add_argument("--debug", default=True)
#     args = parser.parse_args()
#     main(args)
