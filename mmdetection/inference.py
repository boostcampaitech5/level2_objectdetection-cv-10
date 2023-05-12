import mmcv
from mmcv import Config
import argparse
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
from train import find_work_dir

def inf_init_config(cfg, args):
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    root=args.data_dir

    
    # dataset config 수정
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
    cfg.data.test.test_mode = True

    cfg.data.samples_per_gpu = 4

    cfg.seed=2021
    cfg.gpu_ids = [1]
    cfg.work_dir = find_work_dir(args.config_dir, is_train=True)

    # when model use roi-head
    if "roi_head" in cfg.model.keys():
        cfg.model.roi_head.bbox_head.num_classes = 10
        cfg.model.roi_head.bbox_head.loss_cls.type = args.loss_cls
        cfg.model.roi_head.bbox_head.loss_bbox.type = args.loss_bbox
    else:
        # when model not use roi-head
        cfg.model.bbox_head.num_classes = 10
        cfg.model.bbox_head.loss_cls.type = args.loss_cls
        cfg.model.bbox_head.loss_bbox.type = args.loss_bbox

    if "rpn_head" in cfg.model.keys():
        cfg.model.rpn_head.loss_cls.type = args.loss_cls
        cfg.model.rpn_head.loss_bbox.type = args.loss_bbox

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None
    
    return cfg

def inference(args):
    # config file 들고오기
    inference_cfg = Config.fromfile(args.config_dir)
    print("import " + args.model + "for inference")
    cfg = inf_init_config(inference_cfg, args)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    epoch = 'latest'

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산
    
    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
            
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])


    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 1)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--loss_cls', type=str, default='CrossEntropyLoss', help='classification loss')
    parser.add_argument('--loss_bbox', type=str, default='SmoothL1Loss', help='classification loss')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'dataset/'))
    parser.add_argument('--config_dir', type=str, default="./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py")
    parser.add_argument('--wandb_proj', type=str, default='TrashDetection', help='wandb project name')

    args = parser.parse_args()

    print(args)

    inference(args)