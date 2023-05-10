import torch
import wandb
import argparse
import os
import numpy as np
import random
from importlib import import_module
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

import json
import urllib.request
import time
# custom library
from model import *

def alarm():
    payload = {"text": "train finished"}
    # url = "https://hooks.slack.com/services/XXXXXXXXXXXXX"
    url = "https://hooks.slack.com/services/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"
    req = urllib.request.Request(url)
    req.add_header('Content-Type', 'application/json')
    urllib.request.urlopen(req, json.dumps(payload))


def find_work_dir(_model):
    """_summary_
        work_dir for saving pth file
    Args:
        _model (_type_): args.model
    """
    _model = _model.split('/')[-1]
    _model = _model.split('_')
    _model = '_'.join(_model[:-1])

    work_dir = "baseline/mmdetection/work_dirs/" + _model + "_trash"
    
    return work_dir

def init_config(cfg, args):
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    root = args.data_dir

    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train.json' # train json 정보
    cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

    cfg.data.samples_per_gpu = 4

    cfg.seed = 2022
    cfg.gpu_ids = [0]
    # 학습된 파일의 저장 위치 
    cfg.work_dir = find_work_dir(args.config_dir)

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()
    return cfg

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(args):
    seed_everything(args.seed)

    ## when use model.py
    # basemodel = BaseModel(args.config_dir, args.model)
    basemodel = Config.fromfile(args.cfg_dir)
    print("import " + args.model)
    cfg = init_config(basemodel.cfg, args)

    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model)
    model.init_weights()

    # train_detector(model, datasets[0], cfg, distributed=False, validate=False)
    # alarm()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'dataset/'))
    parser.add_argument('--config_dir', type=str, default="./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py")
    parser.add_argument('--wandb_proj', type=str, default='wandb_test', help='wandb project name')

    args = parser.parse_args()


    ## for log in wandb
    # wandb.init(project=args.wandb_proj,
    #            name = "{name for experiment}")
    # wandb.config.update(args)
    print(args)

    train(args)