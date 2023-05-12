import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# load json: modify the path to your own ‘train.json’ file
annotation = "../../dataset/train.json"

with open(annotation) as f: dataset = json.load(f)
print(dataset.keys())
var = [(ann['image_id'], ann['category_id']) for ann in dataset['annotations']]
X = np.ones((len(dataset['annotations']),1))
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
    print("TRAIN:", groups[train_idx])
    print(" ", y[train_idx])
    print(" TEST:", groups[val_idx])
    print(" ", y[val_idx])

    train_img_ids = groups[train_idx]
    val_img_ids = groups[val_idx]
    # train/val 데이터셋 생성
    train_data = {'annotations': [], 'images': [], 'categories': dataset['categories']}
    val_data = {'annotations': [], 'images': [], 'categories': dataset['categories']}

    for data in dataset['images']:
        if data['id'] in train_img_ids:
            train_data['images'].append(data)
        elif data['id'] in val_img_ids:
            val_data['images'].append(data)

    for data in dataset['annotations']:
        if data['image_id'] in train_img_ids:
            train_data['annotations'].append(data)
        elif data['image_id'] in val_img_ids:
            val_data['annotations'].append(data)

    # train/val 데이터셋을 json 파일로 저장
    with open(f'../../dataset/train_split_{i}.json', 'w') as f:
        json.dump(train_data, f)

    with open(f'../../dataset/val_split_{i}.json', 'w') as f:
        json.dump(val_data, f)
