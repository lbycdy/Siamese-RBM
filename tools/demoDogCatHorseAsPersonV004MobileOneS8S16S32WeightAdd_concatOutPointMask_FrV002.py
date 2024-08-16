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
from siamban.models.model_DogCatHorseAsPersonV003MobileOneS8S16S32WeightAdd_concatOutPointMask_FrV002_onlyLoc import ModelBuilder
from siamban.models.backbone.mobileone_stride import reparameterize_model
from siamban.models.backbone.mobileone_strideS16OutTwo import reparameterize_model_allskipscale,reparameterize_model_train
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
args = parser.parse_args()




# args.config = "/home/lbycdy/work/siamban-acm/experiments/siamban_r50_l234/20221119DogCatHorseAsPersonV003MobileOneS8S16S32WeightAdd_ACMOutPointMaskSIn224_4GPU128.yaml"
# args.snapshot = '/home/lbycdy/Results/20221119DogCatHorseAsPersonV003MobileOneS8S16S32WeightAdd_ACMOutPointMaskSIn224_4GPU128/checkpoint_e12_param.pth'

args.config = "/home/lbycdy/work/siamban/experiments/siamban_r50_l234/lb20230301DogCatHorseAsPersonV003MobileOneS8S16S32Add_concatPointMaskDV3VIRandPNMod1TYPECONTEXT0SFTMod0SIn160datasetid2_4GPU127.yaml"
args.snapshot1 = '/home/lbycdy/checkpoint/lb20230301DogCatHorseAsPersonV003MobileOneS8S16S32Add_concatPointMaskDV3VIRandPNMod1TYPECONTEXT0SFTMod0SIn160datasetid2_4GPU127$/checkpoint_e10_param.pth'
args.snapshot2 = '/home/lbycdy/checkpoint/lb20230301DogCatHorseAsPersonV003MobileOneS8S16S32Add_concatPointMaskDV3VIRandPNMod1TYPECONTEXT0SFTMod0SIn160datasetid2_4GPU127$/checkpoint_e7_param.pth'



video_name = "/home/lbycdy/Videos/vlc-record-2023-01-09-16h44m36s-Cat10.mp4-.mp4"
# video_name = "/home/lbycdy/datasets/DogCatHorseVideo/Video1103/Cat/Cat10.mp4"
# video_name = "/home/lbycdy/datasets/DogCatHorseVideo/Video1103/Horse/Horse05.mp4"


# video_name = 0

videogap = 3
args.video_name = video_name
param1 =torch.load(args.snapshot1,map_location=torch.device('cpu'))
for key in param1:
    print(key,param1[key].min(),param1[key].max())
param2 = torch.load(args.snapshot2, map_location=torch.device('cpu'))
for key in param2:
    print(key,param2[key].min(),param2[key].max())
# param =
def get_frames(video_name, videogap):
    if not video_name:

        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)

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
            for i in range(videogap):
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


def getnetinput(image,center_pos,size,flagsearch=False):

    im_h, im_w = image.shape[:2]
    cx,cy = center_pos
    bw,bh = size
    x1 = cx-bw/2
    x2 = x1 + bw
    y1 = cy - bh/2
    y2 = y1 + bh
    cropw = bw*cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
    croph = bh*cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
    if cfg.DATASET.CATDOGHORSETK.FLAG_TEMPLATESEARCH_SIZEFROMCONFIG:
        if flagsearch:
            scale = float(cfg.TRAIN.SEARCH_SIZE) / float(cfg.TRAIN.EXEMPLAR_SIZE)
            cropw *= scale
            croph *= scale
    print(cropw,bw,"crop",flagsearch,cfg.DATASET.CATDOGHORSETK.FLAG_TEMPLATESEARCH_SIZEFROMCONFIG)
    xmin = int(cx - cropw / 2)
    xmax = int(cx + cropw / 2)
    ymin = int(cy - croph / 2)
    ymax = int(cy + croph / 2)
    cropbox = [xmin, ymin, xmax, ymax]

    x1 -= xmin
    x2 -= xmin
    y1 -= ymin
    y2 -= ymin
    padleft = 0 if xmin > 0 else -xmin
    padtop = 0 if ymin > 0 else -ymin
    padright = 0 if xmax < im_w else xmax - im_w
    padbottom = 0 if ymax < im_h else ymax - im_h
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(xmax, im_w)
    ymax = min(ymax, im_h)
    imgcrop = image[ymin:ymax, xmin:xmax, :]
    img_pad = cv2.copyMakeBorder(imgcrop, padtop, padbottom, padleft, padright, cv2.BORDER_CONSTANT,
                                 value=(117, 117, 117))
    h1,w1 = img_pad.shape[:2]
    if cfg.DATASET.CATDOGHORSETK.FLAG_TEMPLATESEARCH_SIZEFROMCONFIG:
        if flagsearch:
            im_patch = cv2.resize(img_pad, (cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE))
            sx = float(cfg.TRAIN.SEARCH_SIZE) / w1
            sy = float(cfg.TRAIN.SEARCH_SIZE) / h1
        else:
            im_patch = cv2.resize(img_pad, (cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE))
            sx = float(cfg.TRAIN.EXEMPLAR_SIZE) / w1
            sy = float(cfg.TRAIN.EXEMPLAR_SIZE) / h1
    else:
        im_patch = cv2.resize(img_pad,(cfg.TRAIN.SEARCH_SIZE,cfg.TRAIN.SEARCH_SIZE))
        sx = float(cfg.TRAIN.SEARCH_SIZE)/w1
        sy = float(cfg.TRAIN.SEARCH_SIZE)/h1
    imgshow = im_patch.copy()
    if flagsearch:
        cv2.namedWindow("imgsearch",cv2.NORM_MINMAX)
        cv2.imshow("imgsearch",imgshow)
    else:
        cv2.namedWindow("imgt", cv2.NORM_MINMAX)
        cv2.imshow("imgt", imgshow)
    im_patch = im_patch.transpose(2, 0, 1)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)
    im_patch = torch.from_numpy(im_patch)
    if cfg.CUDA:
        im_patch = im_patch.cuda()
    return im_patch,sx,sy,imgshow,cropbox
def getnetinput1(image,center_pos,size,flagsearch):

    im_h, im_w = image.shape[:2]
    cx,cy = center_pos
    bw,bh = size
    w_z = bw + 0.5 * (bw + bh)
    h_z = bh + 0.5 * (bw + bh)
    s_z = round(math.sqrt(w_z * h_z))

    cropw = croph = s_z
    xmin = int(cx - cropw / 2)
    xmax = int(cx + cropw / 2)
    ymin = int(cy - croph / 2)
    ymax = int(cy + croph / 2)
    cropbox = [xmin, ymin, xmax, ymax]

    padleft = 0 if xmin > 0 else -xmin
    padtop = 0 if ymin > 0 else -ymin
    padright = 0 if xmax < im_w else xmax - im_w
    padbottom = 0 if ymax < im_h else ymax - im_h
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(xmax, im_w)
    ymax = min(ymax, im_h)
    imgcrop = image[ymin:ymax, xmin:xmax, :]
    img_pad = cv2.copyMakeBorder(imgcrop, padtop, padbottom, padleft, padright, cv2.BORDER_CONSTANT,
                                 value=(117, 117, 117))
    h1,w1 = img_pad.shape[:2]
    if cfg.DATASET.CATDOGHORSETK.FLAG_TEMPLATESEARCH_SIZEFROMCONFIG:
        if flagsearch:
            im_patch = cv2.resize(img_pad, (cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE))
            sx = float(cfg.TRAIN.SEARCH_SIZE) / w1
            sy = float(cfg.TRAIN.SEARCH_SIZE) / h1
        else:
            im_patch = cv2.resize(img_pad, (cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE))
            sx = float(cfg.TRAIN.EXEMPLAR_SIZE) / w1
            sy = float(cfg.TRAIN.EXEMPLAR_SIZE) / h1
    else:
        im_patch = cv2.resize(img_pad,(cfg.TRAIN.SEARCH_SIZE,cfg.TRAIN.SEARCH_SIZE))
        sx = float(cfg.TRAIN.SEARCH_SIZE)/w1
        sy = float(cfg.TRAIN.SEARCH_SIZE)/h1
    imgshow = im_patch.copy()
    if flagsearch:
        cv2.namedWindow("imgsearch",cv2.NORM_MINMAX)
        cv2.imshow("imgsearch",imgshow)
    else:
        cv2.namedWindow("imgt", cv2.NORM_MINMAX)
        cv2.imshow("imgt", imgshow)
    im_patch = im_patch.transpose(2, 0, 1)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)
    im_patch = torch.from_numpy(im_patch)
    if cfg.CUDA:
        im_patch = im_patch.cuda()
    return im_patch,sx,sy,imgshow,cropbox
def generate_points(stride, size):
    ori = - (size // 2) * stride
    x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                       [ori + stride * dy for dy in np.arange(0, size)])
    points = np.zeros((size * size, 2), dtype=np.float32)
    points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

    return points

def _convert_bbox(delta, point):
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
    delta = delta.detach().cpu().numpy()

    delta[0, :] = point[:, 0] - delta[0, :]
    delta[1, :] = point[:, 1] - delta[1, :]
    delta[2, :] = point[:, 0] + delta[2, :]
    delta[3, :] = point[:, 1] + delta[3, :]
    delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
    return delta

def _convert_score(score):
    # if self.cls_out_channels == 1:
    #     score = score.permute(1, 2, 3, 0).contiguous().view(-1)
    #     score = score.sigmoid().detach().cpu().numpy()
    # else:
    score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
    if cfg.TRAIN.FLAG_SIGMOID_LOSS:
        score = score.sigmoid()
        score = score.detach()[:, 1].cpu().numpy()
    else:
        score = score.softmax(1).detach()[:, 1].cpu().numpy()
    return score

def _bbox_clip(cx, cy, width, height, boundary):
    cx = max(0, min(cx, boundary[1]))
    cy = max(0, min(cy, boundary[0]))
    width = max(10, min(width, boundary[1]))
    height = max(10, min(height, boundary[0]))
    return cx, cy, width, height

def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.TRACK.PENALTY_K = 0.05
    cfg.TRACK.WINDOW_INFLUENCE = 0.4
    cfg.TRACK.LR = 0.85

    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model1 = ModelBuilder(cfg)
    model2 = ModelBuilder(cfg)
    if cfg.BACKBONE.TYPE in ["mobileones16outtwo", "mobileones8s16outtwo"]:
        if cfg.TRAIN.MODE_REPARAMETERIZE == 0:
            model1.backbone = reparameterize_model_allskipscale(model1.backbone)
            model2.backbone = reparameterize_model_allskipscale(model2.backbone)
        else:
            model1.backbone = reparameterize_model_train(model1.backbone)
            model2.backbone = reparameterize_model_train(model2.backbone)
    else:
        model1.backbone = reparameterize_model(model1.backbone)
        model2.backbone = reparameterize_model(model2.backbone)
    # load model
    model1.load_state_dict(torch.load(args.snapshot1,
                                      map_location=lambda storage, loc: storage.cpu()))
    model1.eval().to(device)

    model2.load_state_dict(torch.load(args.snapshot2,
                                      map_location=lambda storage, loc: storage.cpu()))
    model2.eval().to(device)

    # build tracker
    points = generate_points(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE)
    hanning = np.hanning(cfg.TRAIN.OUTPUT_SIZE)
    window = np.outer(hanning, hanning)
    window = window.flatten()
    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(args.video_name, videogap):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                pass
            if sum(init_rect) != 0:
                print("init_rect:", init_rect)
                # init_rect = [0,0,frame.shape[1],frame.shape[0]]
                center_pos1 = np.array([init_rect[0] + (init_rect[2] - 1) / 2,
                                        init_rect[1] + (init_rect[3] - 1) / 2])
                center_pos2 = np.array([init_rect[0] + (init_rect[2] - 1) / 2,
                                        init_rect[1] + (init_rect[3] - 1) / 2])
                size1 = np.array([init_rect[2], init_rect[3]])
                size2 = np.array([init_rect[2], init_rect[3]])
                z_crop, _, _, _, _ = getnetinput(frame, center_pos1, size1, flagsearch=False)
                print(z_crop.size(), ":zcrop")
                model1.template(z_crop)
                model2.template(z_crop)
                first_frame = False
        else:
            if cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 4:
                x_crop1, scalex1, scaley1, imgshow1, x_croproi1 = getnetinput1(frame, center_pos1, size1,
                                                                               flagsearch=True)
                x_crop2, scalex2, scaley2, imgshow2, x_croproi2 = getnetinput1(frame, center_pos2, size2,
                                                                               flagsearch=True)
            else:
                x_crop1, scalex1, scaley1, imgshow1, x_croproi1 = getnetinput(frame, center_pos1, size1,
                                                                              flagsearch=True)
                x_crop2, scalex2, scaley2, imgshow2, x_croproi2 = getnetinput(frame, center_pos2, size2,
                                                                              flagsearch=True)
            cropleft1, croptop1 = x_croproi1[:2]
            cropleft2, croptop2 = x_croproi2[:2]
            loc1 = model1.track(x_crop1)
            model1.template(x_crop1)
            loc2 = model2.track(x_crop2)
            model2.template(x_crop2)

            cv2.namedWindow("imgshow1", cv2.NORM_MINMAX)
            cv2.imshow("imgshow1", imgshow1)
            cv2.namedWindow("imgshow2", cv2.NORM_MINMAX)
            cv2.imshow("imgshow2", imgshow2)
            loc1 = loc1.data.cpu().numpy().squeeze()
            loc2 = loc2.data.cpu().numpy().squeeze()
            prior_width = 1.0
            prior_height = 1.0
            prior_variance = [0.1, 0.1, 0.2, 0.2]
            prior_center_x = 0.5
            prior_center_y = 0.5
            bbox1_center_x = loc1[0] * prior_variance[0] * prior_width + prior_center_x
            bbox1_center_y = loc1[1] * prior_variance[1] * prior_height + prior_center_y
            bbox1_width = math.exp(loc1[2] * prior_variance[2]) * prior_width
            bbox1_height = math.exp(loc1[3] * prior_variance[3]) * prior_height
            bbox2_center_x = loc2[0] * prior_variance[0] * prior_width + prior_center_x
            bbox2_center_y = loc2[1] * prior_variance[1] * prior_height + prior_center_y
            bbox2_width = math.exp(loc1[2] * prior_variance[2]) * prior_width
            bbox2_height = math.exp(loc1[3] * prior_variance[3]) * prior_height

            bbox1 = [bbox1_center_x * cfg.TRAIN.SEARCH_SIZE, bbox1_center_y * cfg.TRAIN.SEARCH_SIZE,
                    bbox1_width * cfg.TRAIN.SEARCH_SIZE, bbox1_height * cfg.TRAIN.SEARCH_SIZE]
            bbox2 = [bbox2_center_x * cfg.TRAIN.SEARCH_SIZE, bbox2_center_y * cfg.TRAIN.SEARCH_SIZE,
                     bbox2_width * cfg.TRAIN.SEARCH_SIZE, bbox2_height * cfg.TRAIN.SEARCH_SIZE]

            bbox1[0] /= scalex1
            bbox1[2] /= scalex1
            bbox1[1] /= scaley1
            bbox1[3] /= scaley1
            bbox2[0] /= scalex2
            bbox2[2] /= scalex2
            bbox2[1] /= scaley2
            bbox2[3] /= scaley2

            cx1 = cropleft1 + bbox1[0]
            cy1 = croptop1 + bbox1[1]
            cx2 = cropleft2 + bbox2[0]
            cy2 = croptop2 + bbox2[1]
            # cx1 = bbox1[0] + center_pos1[0] - size1[0] * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR // 2
            # cy1 = bbox1[1] + center_pos1[1] - size1[1] * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR // 2
            # cx2 = bbox2[0] + center_pos2[0] - size2[0] * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR // 2
            # cy2 = bbox2[1] + center_pos2[1] - size2[1] * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR // 2
            # cx = bbox[0] + center_pos[0]
            # cy = bbox[1] + center_pos[1]
            lr = 1.0
            # smooth bbox
            width1 = size1[0] * (1 - lr) + bbox1[2] * lr
            height1 = size1[1] * (1 - lr) + bbox1[3] * lr
            width2 = size2[0] * (1 - lr) + bbox2[2] * lr
            height2 = size2[1] * (1 - lr) + bbox2[3] * lr

            # clip boundary
            cx1, cy1, width1, height1 = _bbox_clip(cx1, cy1, width1,
                                                   height1, frame.shape[:2])
            cx2, cy2, width2, height2 = _bbox_clip(cx2, cy2, width2,
                                                   height2, frame.shape[:2])

            # udpate state
            center_pos1 = np.array([cx1, cy1])
            size1 = np.array([width1, height1])
            center_pos2 = np.array([cx2, cy2])
            size2 = np.array([width2, height2])

            bbox1 = [cx1 - width1 / 2,
                     cy1 - height1 / 2,
                     width1,
                     height1]
            bbox2 = [cx2 - width2 / 2,
                     cy2 - height2 / 2,
                     width2,
                     height2]

            # z_crop, _, _, _ = getnetinput(frame, center_pos, size, flagsearch=False)
            # model.template(z_crop)

            bbox1 = list(map(int, bbox1))
            bbox2 = list(map(int, bbox2))
            cx1 = int(bbox1[0] + bbox1[2] / 2)
            cy1 = int(bbox1[1] + bbox1[3] / 2)
            cx2 = int(bbox2[0] + bbox2[2] / 2)
            cy2 = int(bbox2[1] + bbox2[3] / 2)

            cv2.putText(frame, "%f" % 1, (int(cx1), int(cy1)), cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                        color=(0, 0, 255), thickness=1)
            cv2.putText(frame, "pre", (int(bbox1[0]), int(bbox1[1])), cv2.FONT_HERSHEY_COMPLEX, fontScale=3,
                        color=(0, 124, 234), thickness=1)
            cv2.putText(frame, "%f" % 1, (int(cx2), int(cy2)), cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                        color=(0, 0, 255), thickness=1)

            cv2.rectangle(frame, (bbox1[0], bbox1[1]),
                          (bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]),
                          (0, 124, 234), 3)
            cv2.rectangle(frame, (bbox2[0], bbox2[1]),
                          (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]),
                          (0, 255, 0), 3)
            # tracker.init(frame, bbox)

            # colorlist = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            # for i in range(pred_bbox_good.shape[1]):
            #     cx, cy, w, h = pred_bbox_good[:, i]
            #     xmin = cx - w / 2
            #     ymin = cy - h / 2
            #     xmax = cx + w / 2
            #     ymax = cy + h / 2
            #     xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
            #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
            #                   colorlist[i % 5], 3)

            cv2.imshow(video_name, frame)
            key = cv2.waitKey(0)
            if key == 27:
                exit()


if __name__ == '__main__':
    main()
