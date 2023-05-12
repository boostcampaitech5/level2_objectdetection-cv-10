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


def find_work_dir(_model, is_train=True):
    """_summary_
        work_dir for saving pth file
    Args:
        _model (_type_): args.model
    """
    _model = _model.split('/')[-1]
    _model = _model.split('_')
    _model = '_'.join(_model[:-1])

    if is_train:
        work_dir = "baseline/mmdetection/work_dirs/" + _model + "_trash"
    else:
        work_dir = "baseline/mmdetection/work_dirs/" + _model + "_implement"
    return work_dir

def init_config(cfg, args):
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    root = args.data_dir

    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train_split_0.json' # train json 정보
    cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = root + 'val_split_0.json' # valid json 정보

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

    cfg.data.samples_per_gpu = 4

    cfg.seed = 2022
    cfg.gpu_ids = [0]

    # 학습된 파일의 저장 위치 
    cfg.work_dir = find_work_dir(args.config_dir)

    # 최종 분류기 
    # when model use roi-head
    if "roi_head" in cfg.model.keys():
        cfg.model.roi_head.bbox_head.num_classes = 10
        cfg.model.roi_head.loss_cls.type = args.loss_cls
        cfg.model.roi_head.loss_bbox.type = args.loss_bbox
    else:
        # when model not use roi-head
        cfg.model.bbox_head.num_classes = 10
        cfg.model.bbox_head.loss_cls.type = args.loss_cls
        cfg.model.bbox_head.loss_bbox.type = args.loss_bbox

    if "rpn_head" in cfg.model.keys():
        cfg.model.rpn_head.loss_cls.type = args.loss_cls
        cfg.model.rpn_head.loss_bbox.type = args.loss_bbox
        

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.runner.max_epochs = args.epochs

    cfg.workflow = [('train', 1), ('val', 1)]

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
    # basemodel = getattr(import_module("model"),args.model)
    # basemodel = basemodel()
    # cfg = init_config(basemodel.cfg, args)

    basemodel_cfg = Config.fromfile(args.config_dir)
    print("import " + args.model)
    cfg = init_config(basemodel_cfg, args)

    # cfg.log_config.hooks = [
    #     dict(type='TextLoggerHook'),
    #     dict(type='MMDetWandbHook',
    #         init_kwargs={'project': 'mmdetection'},
    #         interval=10,
    #         log_checkpoint=True,
    #         log_checkpoint_metadata=True,
    #         num_eval_images=100,
    #         bbox_score_thr=0.3)]

    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model)
    model.init_weights()

    train_detector(model, datasets[0], cfg, distributed=False, validate=True)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 1)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--loss_cls', type=str, default='FocalLoss', help='classification loss')
    parser.add_argument('--loss_bbox', type=str, default='SmoothL1Loss', help='classification loss')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'dataset/'))
    parser.add_argument('--config_dir', type=str, default="./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py")
    parser.add_argument('--wandb_proj', type=str, default='TrashDetection', help='wandb project name')

    args = parser.parse_args()


    # for log in wandb
    # wandb.init(project=args.wandb_proj,
    #            name = f"{args.config_dir.split('/')[-1].split('.')[0]}")
    # wandb.config.update(args)
    
    print(args)

    train(args)