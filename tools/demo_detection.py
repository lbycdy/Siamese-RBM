from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from siamban.core.config import cfg
from siamban.models.model_DogCatHorseAsPersonV003MobileOneS8S16S32WeightAdd_ACMOutPointMask_FrV002_onlyLoc import ModelBuilder
from siamban.models.backbone.mobileone_stride import reparameterize_model
from siamban.models.backbone.mobileone_strideS16OutTwo import reparameterize_model_allskipscale,reparameterize_model_train
from siamban_detection import SiamBANTracker
import torch.nn.functional as F
import numpy as np
import math
from siamban.utils.bbox import corner2center, \
        Center, center2corner, Corner
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('save',action='store_true',help='whether visualzie resule')
args = parser.parse_args()




args.config = "/home/lbycdy/work/siamban/experiments/siamban_r50_l234/20221201DogCatHorseAsPersonV003MobileOneS8S16S32WeightAdd_ACMOutPointMaskVidImgRandSIn224.yaml"
args.snapshot = '/home/lbycdy/checkpoint/checkpoint_e20.pth'


video_name = "/home/lbycdy/datasets/DogCatHorseVideo/Video1103/Dog/Dog03.mp4"


# video_name = 0

args.video_name = video_name
param =torch.load(args.snapshot,map_location=torch.device('cpu'))
for key in param['state_dict']:
    print(key,param['state_dict'][key].min(),param['state_dict'][key].max())
# param =
def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4') or video_name.endswith('MP4') or video_name.endswith('LRV'):
        cap = cv2.VideoCapture(args.video_name)
        # cap.set(cv2.CAP_PROP_POS_FRAMES,25*9)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder(cfg)

    model.backbone = reparameterize_model_allskipscale(model.backbone)

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu())['state_dict'])
    model.eval().to(device)

    # build tracker
    tracker = SiamBANTracker(model,cfg = cfg)
    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(args.video_name):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            outputs = tracker.detection(frame,init_rect)
            first_frame = False
            pred_boxes = outputs["pred_bbox"][0].bbox.tolist()
            extra = outputs["pred_bbox"][0].extra_fields
            labels = extra["labels"]
            print('labels',labels)
            scores = extra["scores"]
            print('scores',scores)
            detection_img = outputs['d_img']
            cv2.imshow("d_img", detection_img)
            color = [(255, 0, 0), (0,255,0), (0,0,255)]
            if len(pred_boxes)!=0:
                for i in range(len(pred_boxes)):
                    d = pred_boxes[i]
                    print(d)
                    bbox = list(map(int, d))
                    cv2.rectangle(detection_img, (bbox[0], bbox[1]),
                                  (bbox[2], bbox[3]),
                                  color[i], 3)
                cv2.namedWindow("detection_image", cv2.NORM_HAMMING)
                cv2.imshow("detection_image", detection_img)
                key = cv2.waitKey()
                if key == 27:
                    exit()
            cv2.imshow(video_name, frame)
            key = cv2.waitKey(0)
            if key == 27:
                exit()
        else:
            outputs = tracker.detection(frame, init_rect)
            pred_boxes = outputs["pred_bbox"][0].bbox.tolist()
            extra = outputs["pred_bbox"][0].extra_fields
            labels = extra["labels"]
            print('labels', labels)
            scores = extra["scores"]
            print('scores', scores)
            detection_img = outputs['d_img']
            cv2.imshow("d_img", detection_img)
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            if len(pred_boxes) != 0:
                for i in range(len(pred_boxes)):
                    d = pred_boxes[i]
                    print(d)
                    bbox = list(map(int, d))
                    cv2.rectangle(detection_img, (bbox[0], bbox[1]),
                                  (bbox[2], bbox[3]),
                                  color[i], 3)
                cv2.namedWindow("detection_image", cv2.NORM_HAMMING)
                cv2.imshow("detection_image", detection_img)
                key = cv2.waitKey()
                if key == 27:
                    exit()
            cv2.imshow(video_name, frame)
            key = cv2.waitKey(0)
            if key == 27:
                exit()



if __name__ == '__main__':
    main()
