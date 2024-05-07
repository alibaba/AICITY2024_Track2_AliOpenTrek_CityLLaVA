import os, json
from tqdm import tqdm

image_path = './data'
data_path = './processed_anno/llava_format/wts_bdd_llava_qa_train_stage_filted.json'
save_path = './processed_anno/llava_format/wts_bdd_llava_qa_train_stage_filted_checked.json'
save_miss_path = './processed_anno/llava_format/wts_bdd_llava_qa_train_stage_filted_miss.json'
with open(data_path, 'r') as f:
    data_json = json.load(f)
print(f'num:{len(data_json)}')
miss_data = []
new_data = []
for data in tqdm(data_json):
    if 'image' in data.keys():
        sample_path = os.path.join(image_path, data['image'])
    if not os.path.exists(sample_path):
        print(sample_path)
        miss_data.append(sample_path)
    else:
        new_data.append(data)

print(f'{len(data_json)} vs {len(new_data)}')

with open(save_path, 'w') as f:
    f.write(json.dumps(new_data, indent=2, ensure_ascii=False))

with open(save_miss_path, 'w') as f1:
    f1.write(json.dumps(miss_data, indent=2, ensure_ascii=False))


