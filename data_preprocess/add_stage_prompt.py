import os, json, glob
from tqdm import tqdm
import random
import numpy as np

def random_shuffle_conversations(conversations):
    question_nums = len(conversations)
    assert question_nums % 2 == 0, 'Pairs incomplete'

    indices = np.arange(question_nums).reshape(-1, 2).tolist()
    random.shuffle(indices)
    indices = np.asarray(indices).reshape(-1).tolist()
    shuffled_conversations = list()
    for ind in indices:
        shuffled_conversations.append(conversations[ind])
    return shuffled_conversations


if __name__ == '__main__':
    labels_path = './data/processed_anno/llava_format/wts_bdd_llava_qa_train.json'
    save_path = './data/processed_anno/llava_format/wts_bdd_llava_qa_train_stage.json'


    # wts_prompt = "<image>\nThis is an image in '{viewpoint}' stage. Pay attention to the pedestrian in the green bounding box and the vehicle in the blue bounding box. Note that the bounding box may not exist, then answer the following questions:\n{question}"
    # bdd_prompt = "<image>\nThis is an image in '{viewpoint}' stage. Pay attention to the pedestrian in the green bounding box and the ego-vehicle. Note that the bounding box may not exist, then answer the following questions:\n{question}"

    wts_prompt = "<image>\nThis is an image in '{viewpoint}' stage. {question}"
    bdd_prompt = "<image>\nThis is an image in '{viewpoint}' stage. {question}"


    data_json = json.load(open(labels_path))
    for data in tqdm(data_json):
        data['conversations'] = random_shuffle_conversations(data['conversations'])
        viewpoint = data['image'].split('/')[-1].split('_')[-1].split('.')[0]
        if 'BDD_PC_5k' in data['image'] or 'vehicle_view' in data['image']:
            prompt = bdd_prompt
        else:
            prompt = wts_prompt
        for i, cc in enumerate(data['conversations']):
            if cc['from'] == 'human':
                cc['value'] = cc['value'].replace('<image>\n', '').replace('the green box', 'the green bounding box').replace('the blue box', 'the blue bounding box')
                if 'BDD_PC_5k' in data['image'] or 'vehicle_view' in data['image']:
                    cc['value'] = cc['value'].replace('the vehicle with the blue bounding box', 'the ego-vehicle').replace('the vehicle with the blue box', 'the ego-vehicle')
                if i == 0:
                    cc['value'] = prompt.format(viewpoint = viewpoint, question = cc['value'])

    print(len(data_json))
    with open(save_path, 'w') as f:
        f.write(json.dumps(data_json, indent=2, ensure_ascii=False))  