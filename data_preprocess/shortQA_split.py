from http import HTTPStatus
import json
import tqdm
import multiprocessing
import os
import argparse

import dashscope
from openai import OpenAI

# Short QA Construction
# 1) utilizes llm to categorize each sentence of the descriptions into predefined dimensions.


# * here we use qwen (dashscope api) instead of gpt-4 or chat-gpt, you can easily change it with openAI api
# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
def call_with_messages(content, model_type, key):
    if model_type == 'Qwen':
        dashscope.api_key = key
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': content
                    }]

        response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_plus,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
        )
        if response.status_code == HTTPStatus.OK:
            # print(response)
            response = response.output['choices'][0]['message']['content']
            print(response)
            return response

        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            return None
        
    elif model_type == 'Openai':
        client = OpenAI(api_key = key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model="gpt-4",        
        )
        response = chat_completion.choices[0].message.content
        return response


def classify_single_caption(caption_text, caption_type, model_type, api_key):
    # 将长描述按照句号进行拆分
    caption_text += ' '
    caption_list = caption_text.split('. ')

    for i, c in enumerate(caption_list):
        if len(caption_list[i]) > 0:
            caption_list[i] = "{}.{}.".format(str(i + 1), c)
    new_caption_text = '\n'.join(caption_list)
    # print(new_caption_text)

    if caption_type == "pedestrian":
        content = ("Please select the most appropriate label for each descriptive text from the following options, and format the output by providing the text index followed by the letter a, b, c, d, or e. Each selection should be on a new line.\n"
                    "Option a. Description of the pedestrian's age, height and clothing.\n"
                    "Option b. Description of the orientation and relative position relationship between pedestrians and vehicles.\n"
                    "Option c. Description of the pedestrian's line of sight direction and movement status. \n"
                    "Option d. Pedestrians' surrounding environment, weather conditions, and road conditions. \n"
                    "Option e. A description of whether pedestrians have potential risks or accidents, summarizing the pedestrian situation. \n"
                   "here is the descriptive text: \n" + new_caption_text)
        # Previous versions used
        content_cn = ("请为每一条描述文本分别从以下选项中挑选一个最符合的主题，最终按行输出文本序号和abcde其中一个字母。\n"
                    "选项a.行人的年龄、身高等基本特征和穿着描述。\n"
                    "选项b.行人的与车辆的朝向、相对位置关系的描述。\n"
                    "选项c.行人的视线方向和运动状态。\n"
                    "选项d.行人所处周边环境情况、天气情况、道路情况。\n"
                    "选项e.行人是否存在潜在风险或是否发生事故的相关描述，对行人情况的总结概括。\n"
                   "描述文本: \n" + new_caption_text)
    else:
        content = ("Please select the most appropriate label for each descriptive text from the following options, and format the output by providing the text index followed by the letter a, b, c, d, or e. Each selection should be on a new line.\n"
                    "Option a. Description of the orientation and relative position relationship between pedestrians and vehicles. \n"
                    "Option b. Description of the vehicle's driving status and speed. \n"
                    "Option c. Description of the pedestrian' age, height and clothing. \n"
                    "Option d. Description of the surrounding environment, weather conditions, and road conditions. \n"
                    "Option e. A summary of the vehicle's situation.\n"
                   "here is the descriptive text: \n" + new_caption_text)
        # Previous versions used
        content_cn = ("请为每一条描述文本分别从以下选项中挑选一个最符合的主题，最终按行输出文本序号和abcd其中一个字母。\n"
                   "选项a.车辆与行人的相对位置关系等描述\n"
                    "选项b.车辆的行驶状态、速度的描述。\n"
                    "选项c.对行人的年龄、身高等基本特征和穿着描述。\n"
                    "选项d.车辆所处周边环境情况、天气情况、道路情况。\n"
                    "选项e.对车辆情况的总结性概括。\n"
                   "描述文本: \n" + new_caption_text)

    print(content)
    response = call_with_messages(content, model_type, api_key)
    return response

def classify_process(input_data_list, save_file, caption_type, model_type, api_key):
    w = open(save_file, 'w', encoding='utf-8')
    for data in tqdm.tqdm(input_data_list):
        id = data['id']
        try:
            conversations = data['conversations']
            pedestrian_caption = conversations[1]['value']
            vehicle_caption = conversations[3]['value']

            if caption_type == "vehicle":
                # 描述拆分
                response = classify_single_caption(vehicle_caption, "vehicle", model_type, api_key)
                data['vehicle_response'] = response

            else:
                response = classify_single_caption(pedestrian_caption, "pedestrian", model_type, api_key)
                data['pedestrian_response'] = response

            w.write(json.dumps(data, ensure_ascii=False) + '\n')
            w.flush()
        except Exception as e:
            print('{}, {}'.format(id, e))

if __name__ == '__main__':
    # test_caption = "The pedestrian is a male in his 10s, with a height of 160 cm. He is wearing a yellow T-shirt and black slacks. It is a weekday in an urban area with clear weather and dark brightness. The road surface is dry and level, made of asphalt. The pedestrian is standing diagonally to the left in front of a moving vehicle, which is far away. His body is perpendicular to the vehicle and to the right. His line of sight indicates that he is crossing the road. He is closely watching his destination while unaware of the vehicle. The pedestrian's speed is slow, and he intends to cross immediately in front of or behind the vehicle. The road he is on is a main road with one-way traffic and two lanes. Sidewalks are present on both sides"
    # classify_single_caption(test_caption, "pedestrian")

    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, default='Qwen', help='Choose your LLM')
    args.add_argument('--api-key', type=str, required=True, help='Your API key for chosen LLM')

    input_file = "./processed_anno/llava_format/wts_bdd_train.json"
    save_file = "./processed_anno/caption_split/caption_split.json"

    if not os.path.exists("./processed_anno/caption_split"):
        os.makedirs("./processed_anno/caption_split")

    num_works = 1
    endata_multiprocess = []
    with open(input_file, 'r') as f:
        input_data_json = json.load(f)
        # input_data_json = input_data_json[0:10]
    for i in range(num_works):
        endata_multiprocess.append([])
    for i in range(len(input_data_json)):
        endata_multiprocess[i%num_works].append(input_data_json[i])

    pool = multiprocessing.Pool(processes=num_works)
    tasks = []
    for i in range(num_works):
        tasks.append((endata_multiprocess[i], save_file.replace(".json", "_{}_{}.json".format("vehicle", i)), "vehicle", args.model, args.api_key))
        tasks.append((endata_multiprocess[i], save_file.replace(".json", "_{}_{}.json".format("pedestrian", i)), "pedestrian", args.model, args.api_key))
    pool.starmap(classify_process, tasks)
    pool.close()
    pool.join()

