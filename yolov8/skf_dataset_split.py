import json
import os
import shutil

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

annotation = "/opt/ml/dataset/train.json"
with open(annotation) as file:
    dataset = json.load(file)

temp_var = [(ann["image_id"], ann["category_id"]) for ann in dataset["annotations"]]
X = np.ones((len(dataset["annotations"]), 1))
y = np.array([var[1] for var in temp_var])
groups = np.array([var[0] for var in temp_var])

root_dir = "/opt/ml/yolo_dataset/"
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)
another_valid_ids = []  # ! coco2yolo.py를 위해 valid indices를 별도로 저장해두는 list

for kf_num, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
    train_img_ids = groups[train_idx]
    valid_img_ids = groups[valid_idx]
    another_valid_ids.append(list(set(valid_img_ids)))

    dir_name = root_dir + f"skf_split_{kf_num}/images/"

    # ! SKF 이후 Train/Valid Images로 분할하여 복사 이동, .json Annotation은 기존 파일 사용
    for a in set(train_img_ids):
        a_name = str(a).zfill(4) + ".jpg"
        train_src = "/opt/ml/dataset/images/train/" + a_name
        train_tar1 = dir_name + "train"
        train_tar2 = os.path.join(train_tar1, a_name)
        shutil.copy(train_src, train_tar2)
    for b in set(valid_img_ids):
        b_name = str(b).zfill(4) + ".jpg"
        valid_src = "/opt/ml/dataset/images/train/" + b_name
        valid_tar1 = dir_name + "valid"
        valid_tar2 = os.path.join(valid_tar1, b_name)
        shutil.copy(valid_src, valid_tar2)


def get_valid_ids():
    return another_valid_ids
