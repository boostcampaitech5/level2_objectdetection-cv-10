import json

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# load json: modify the path to your own ‘train.json’ file
annotation = "/opt/ml/dataset/train.json"

with open(annotation) as file:
    dataset = json.load(file)

temp_var = [(ann["image_id"], ann["category_id"]) for ann in dataset["annotations"]]
X = np.ones((len(dataset["annotations"]), 1))
y = np.array([var[1] for var in temp_var])
groups = np.array([var[0] for var in temp_var])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

for kf_num, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
    print("TRAIN:", groups[train_idx])
    print(" ", y[train_idx])
    print(" TEST:", groups[valid_idx])
    print(" ", y[valid_idx])

    train_img_ids = groups[train_idx]
    valid_img_ids = groups[valid_idx]

    # ! Train/Valid Dataset 생성
    train_data = {"annotations": [], "images": [], "categories": dataset["categories"]}
    valid_data = {"annotations": [], "images": [], "categories": dataset["categories"]}

    for img_data, annot_data in zip(dataset["images"], dataset["annotations"]):
        if img_data["id"] in train_img_ids:
            train_data["images"].append(img_data)
        elif img_data["id"] in valid_img_ids:
            valid_data["images"].append(img_data)

        if annot_data["image_id"] in train_img_ids:
            train_data["annotations"].append(annot_data)
        elif annot_data["image_id"] in valid_img_ids:
            valid_data["annotations"].append(annot_data)

    # ! Train/Valid Dataset을 json 파일로 저장
    with open(f"/opt/ml/dataset/skf_yolo/train_split_{kf_num}.json", "w") as tf:
        json.dump(train_data, tf)

    with open(f"/opt/ml/dataset/skf_yolo/valid_split_{kf_num}.json", "w") as vf:
        json.dump(valid_data, vf)
