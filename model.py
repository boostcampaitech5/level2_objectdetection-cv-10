# 모듈 import
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

class FasterRcnn():
    def __init__(self):
        # config file 들고오기
        self.cfg = Config.fromfile('baseline/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
        print("import FasterRcnn")