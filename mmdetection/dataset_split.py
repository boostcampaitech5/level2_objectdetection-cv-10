# import json
# import random
# from collections import Counter

# def get_most_frequent_number(classes):
#     # classes 리스트에서 각 숫자가 몇 번씩 등장하는지를 카운트합니다.
#     counter = Counter(classes)
#     # 가장 빈번하게 등장하는 숫자와 그 등장 횟수를 찾습니다.
#     most_common = counter.most_common(1)[0]
#     # 가장 빈번하게 등장하는 숫자를 반환합니다.
#     return most_common[0]

# # 데이터셋 파일 경로
# data_file = "dataset/train.json"

# # 클래스별 데이터 분할 비율
# train_ratio = 0.8
# val_ratio = 0.2

# # 데이터셋 로드
# with open(data_file, "r") as f:
#     data = json.load(f)

# # 클래스 정보와 이미지 ID를 모두 담은 리스트 생성
# id_class_list = []
# for img in data["images"]:
#     # 이미지 ID와 해당 이미지에 등장하는 모든 객체의 클래스 정보를 담은 리스트 생성
#     id_class = {"id": img["id"], "classes": []}
#     for ann in data["annotations"]:
#         if ann["image_id"] == img["id"]:
#             id_class["classes"].append(ann["category_id"])
#     id_class_list.append(id_class)

# for id_class in id_class_list:
#     classes = id_class["classes"]
#     id_class_list[id_class["id"]] = get_most_frequent_number(classes)

# # 클래스별로 이미지 ID를 분할
# train_ids = set()
# val_ids = set()
# for class_idx in data["categories"]:
#     # 해당 클래스의 이미지 ID만 추출
#     class_img_ids = set(id_class["id"] for id_class in id_class_list if class_idx["id"] in id_class["classes"])

#     # 분할 비율에 맞게 train/val 이미지 ID를 선택
#     num_train = int(len(class_img_ids) * train_ratio)
#     num_val = int(len(class_img_ids) * val_ratio)
#     train_ids.update(random.sample(class_img_ids, num_train))
#     val_ids.update(random.sample(class_img_ids - train_ids, num_val))

# # train/val 데이터셋 구성
# train_data = []
# val_data = []
# for img in data["images"]:
#     # 이미지 ID가 train에 속하면 train_data에 추가, val에 속하면 val_data에 추가
#     if img["id"] in train_ids:
#         train_data.append(img)
#     elif img["id"] in val_ids:
#         val_data.append(img)
        
# # train/val 데이터셋을 각각의 json 파일로 저장
# train_out = {"images": train_data, "annotations": data["annotations"], "categories": data["categories"]}
# val_out = {"images": val_data, "annotations": data["annotations"], "categories": data["categories"]}
# with open(f"../../dataset/train_split_{train_ratio}.json", "w") as f:
#     json.dump(train_out, f)
# with open(f"../../dataset/val_split_{val_ratio}.json", "w") as f:
#     json.dump(val_out, f)

import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# load json: modify the path to your own ‘train.json’ file
annotation = "dataset/train.json"

with open(annotation) as f: data = json.load(f)

var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
X = np.ones((len(data['annotations']),1))
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

for train_idx, val_idx in cv.split(X, y, groups):
    print("TRAIN:", groups[train_idx])
    print(" ", y[train_idx])
    print(" TEST:", groups[val_idx])
    print(" ", y[val_idx])
    print("Train set : ", sorted(set(groups[train_idx])))
    print("-----------------------")
    print("Train set : ", sorted(set(groups[val_idx])))

