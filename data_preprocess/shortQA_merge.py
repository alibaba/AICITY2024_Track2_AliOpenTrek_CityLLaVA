import json
import tqdm
import copy

# Short QA Construction
# 2）the sentences within the same dimension are concatenated to form a cohesive segment.
# 3）for short QA, some filtering and sampling operations will be carried out to improve data quality and distribution.
# 4) use high-quality prompt after "3.2.3 Textual Prompt Engineering" to construct fullQA dataset.

def load_jsonl(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def shortQA_merge_pedestrian():
    input_file_list = ["./data/processed_anno/caption_split/caption_split_pedestrian_0.json"
                       ]
    save_file = "./data/processed_anno/caption_split/caption_split_pedestrian_merge.json"
    question_list = [
        "Describe the age, height and clothing of the pedestrian in the green box.",
        "Describe the position of the pedestrian in the green box relative to the vehicle.",
        "Describe the line of sight and movement status of the pedestrian in the green box.",
        "Describe the weather conditions and road environment."
    ]

    result = []

    for input_file in input_file_list:
        data_list = load_jsonl(input_file)
        for data in data_list:
            id = data['id']
            conversations = data['conversations']
            pedestrian_caption_ori = conversations[1]['value']

            # 描述拆分
            pedestrian_caption = pedestrian_caption_ori + ' '
            pedestrian_caption_list = pedestrian_caption.split('. ')
            pedestrian_caption_list = [sentence.strip() + '.' for sentence in pedestrian_caption_list if sentence]

            pedestrian_res = [[], [], [], []]
            pedestrian_response = data['pedestrian_response']
            pedestrian_response_split = pedestrian_response.split('\n')
            for pedestrian_response_data in pedestrian_response_split:
                pedestrian_num, pedestrian_class = pedestrian_response_data.split('.')
                pedestrian_num = int(pedestrian_num) - 1
                pedestrian_class = pedestrian_class.lower()
                if pedestrian_num >= len(pedestrian_caption_list):
                    continue
                if 'a' in pedestrian_class:
                    pedestrian_res[0].append(pedestrian_caption_list[pedestrian_num])
                elif 'b' in pedestrian_class:
                    pedestrian_res[1].append(pedestrian_caption_list[pedestrian_num])
                elif 'c' in pedestrian_class:
                    pedestrian_res[2].append(pedestrian_caption_list[pedestrian_num])
                elif 'd' in pedestrian_class:
                    pedestrian_res[3].append(pedestrian_caption_list[pedestrian_num])

            for i in range(len(pedestrian_res)):
                if len(pedestrian_res[i]) == 0:
                    continue
                conversations_new = []
                pedestrian_res[i] = ' '.join(pedestrian_res[i])
                conversations_new.append({
                    "from": "human",
                    "value": "<image>\n" + question_list[i]
                })
                conversations_new.append({
                    "from": "gpt",
                    "value": pedestrian_res[i]
                })
                data['conversations'] = conversations_new
                data['full_flag'] = 0
                data_new = copy.deepcopy(data)
                result.append(data_new)

            conversations_new = []
            conversations_new.append({
                "from": "human",
                "value": "<image>\nThis picture shows the relationship between the pedestrian in the green box and the vehicle in the blue box. Describe the pedestrian in the green box or the pedestrian closest to the vehicle based on age, height, clothing, line of sight, relative position to the vehicle, movement status, weather conditions and road environment."
            })
            conversations_new.append({
                "from": "gpt",
                "value": pedestrian_caption_ori
            })
            data['conversations'] = conversations_new
            data['full_flag'] = 1
            data_new = copy.deepcopy(data)
            result.append(data_new)
    print("len of shortQA_merge_prdestrian data is ", len(result))
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def shortQA_merge_vehicle():
    input_file_list = ["./data/processed_anno/caption_split/caption_split_vehicle_0.json"]
    save_file = "./data/processed_anno/caption_split/caption_split_vehicle_merge.json"
    question_list = [
        "Describe the position of the vehicle in the blue box relative to the pedestrian in the green box.",
        "Describe the driving status of the vehicle in the blue box."
        "Describe the attributes of the pedestrian with the green box.", # not used
        "Describe the weather conditions and road environment." # not used
    ]

    result = []

    for input_file in input_file_list:
        data_list = load_jsonl(input_file)
        for data in data_list:
            id = data['id']
            conversations = data['conversations']
            vehicle_caption_ori = conversations[3]['value']

            # split description
            vehicle_caption = vehicle_caption_ori + ' '
            pedestrian_caption_list = vehicle_caption.split('. ')
            pedestrian_caption_list = [sentence.strip() + '.' for sentence in pedestrian_caption_list if sentence]

            pedestrian_res = [[], [], [], []]
            pedestrian_response = data['vehicle_response']
            pedestrian_response_split = pedestrian_response.split('\n')
            for pedestrian_response_data in pedestrian_response_split:
                pedestrian_num, pedestrian_class = pedestrian_response_data.split('.')
                pedestrian_num = int(pedestrian_num) - 1
                pedestrian_class = pedestrian_class.lower()
                if pedestrian_num >= len(pedestrian_caption_list):
                    # print(pedestrian_response)
                    # print(pedestrian_caption_list)
                    continue
                if 'a' in pedestrian_class:
                    pedestrian_res[0].append(pedestrian_caption_list[pedestrian_num])
                elif 'b' in pedestrian_class:
                    pedestrian_res[1].append(pedestrian_caption_list[pedestrian_num])

            for i in range(len(pedestrian_res)):
                if len(pedestrian_res[i]) == 0:
                    continue
                conversations_new = []
                pedestrian_res[i] = ' '.join(pedestrian_res[i])
                conversations_new.append({
                    "from": "human",
                    "value": "<image>\n" + question_list[i]
                })
                conversations_new.append({
                    "from": "gpt",
                    "value": pedestrian_res[i]
                })
                data['conversations'] = conversations_new
                data['full_flag'] = 0
                data_new = copy.deepcopy(data)
                result.append(data_new)

            # print(conversations_new)
            conversations_new = []
            conversations_new.append({
                "from": "human",
                "value": "<image>\nThis picture shows the relationship between the vehicle in the blue box and the pedestrian in the green box. Describe the vehicle in the blue box or the vehicle closest to the pedestrian based on the relative position to the pedestrian, driving status, weather conditions and road environment. And describe the age, height, clothing of the pedestrian."
            })
            conversations_new.append({
                "from": "gpt",
                "value": vehicle_caption_ori
            })
            data['conversations'] = conversations_new
            data['full_flag'] = 1
            data_new = copy.deepcopy(data)
            result.append(data_new)
    print("len of shortQA_merge_vehicle data is ", len(result))
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def shortQA_merge():
    input_file_list = ["./data/processed_anno/caption_split/caption_split_pedestrian_merge.json",
                       "./data/processed_anno/caption_split/caption_split_vehicle_merge.json"]
    save_file = "./data/processed_anno/caption_split/caption_split_merge.json"
    input_json_merge = []
    for input_file in input_file_list:
        with open(input_file, "r", encoding="utf-8") as f:
            input_json = json.load(f)
            input_json_merge.extend(input_json)
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(input_json_merge, f, ensure_ascii=False, indent=4)



# Split short QA and long QA data, filter and sample for short QA
def data_filter():
    input_file = "./processed_anno/caption_split/caption_split_merge.json"
    save_file = './processed_anno/llava_format/wts_bdd_llava_qa_train.json'

    result_single_question = []
    result_full_question = []
    with open(input_file, "r", encoding="utf-8") as f:
        input_json = json.load(f)
        for data in input_json:
            if data["full_flag"] == 1:
                result_full_question.append(data)
                continue
            else:
                # if there is no pedestrian detection box in the video, filter the QA description of pedestrian
                if "P" not in data["tag"]:
                    question = data["conversations"][0]["value"]
                    if "pedestrian" in question:
                        continue
                # For monitoring perspectives, if there is no vehicle detection box in the video, filter the QA description of the vehicle
                if 'BDD_PC_5k' not in data['image'] and 'vehicle_view' not in data['image']:
                    if "V" not in data["tag"]:
                        question = data["conversations"][0]["value"]
                        if "vehicle" in question:
                            continue
                result_single_question.append(data)
    print("len of result_single_question is ", len(result_single_question))
    print("len of result_full_question is ", len(result_full_question))

    import random
    result_single_question = random.sample(result_single_question, len(result_full_question))

    result_full_question.extend(result_single_question)
    print("len of final_question is ", len(result_full_question))

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(result_full_question, f, ensure_ascii=False, indent=4)

    # not used
    # save_file = input_file.replace('.json', '_shortQA.json')
    # with open(save_file, "w", encoding="utf-8") as f:
    #     json.dump(result_single_question, f, ensure_ascii=False, indent=4)
    #
    # save_file = input_file.replace('.json', '_fullQA.json')
    # with open(save_file, "w", encoding="utf-8") as f:
    #     json.dump(result_full_question, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    shortQA_merge_pedestrian()
    shortQA_merge_vehicle()
    shortQA_merge()
    data_filter()
