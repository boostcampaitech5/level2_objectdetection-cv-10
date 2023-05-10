# 모듈 import
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

class BaseModel():
    def __init__(self, cfg_dir, model_name):
        # config file 들고오기
        self.cfg = Config.fromfile(cfg_dir)
        print("import " + model_name)