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






args.config = "/home/lbycdy/work/siamban/experiments/siamban_r50_l234/lb20230301DogCatHorseAsPersonV003MobileOneS8S16S32Add_concatPointMaskDV3VIRandPNMod1TYPECONTEXT0SFTMod0SIn160datasetid2_4GPU127.yaml"
args.snapshot = '/home/lbycdy/checkpoint/lb20230301DogCatHorseAsPersonV003MobileOneS8S16S32Add_concatPointMaskDV3VIRandPNMod1TYPECONTEXT0SFTMod0SIn160datasetid2_4GPU127$/checkpoint_e5_param.pth'



video_name = "/home/lbycdy/datasets/DogCatHorseVideo/Video1103/Cat/Cat10.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Cat/Cat02.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Cat/Cat03.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Cat/Cat04.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Cat/Cat05.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Cat/Cat06.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Cat/Cat10_start144.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Cat/Cat10_start1630.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Cat/Cat10_start2070.mp4"

# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Cat/Cat10_start144.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Cat/Cat10_start1630.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Cat/Cat10_start2070.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Dog/Dog01.mp4"

# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Dog/Dog04.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Dog/Dog07.MP4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Dog/Dog08.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Dog/Dog11.mp4"

# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Horse/Horse03.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Horse/Horse05.mp4"
# video_name = "/media/ethan/OldDisk/home/ethan/test_videos/DogCatHorseVideo/Video1103/Horse/Horse07.mp4"
# video_name  = "/home/ethan/Downloads/CST HUMAN S16-TEST-1 CAMERA.avi"
# video_name = "/home/ethan/Downloads/mmexport1670059402899.mp4"
# video_name = 0
videogap = 3
args.video_name = video_name
param =torch.load(args.snapshot,map_location=torch.device('cpu'))
if "state_dict" in param:
    param = param['state_dict']
for key in param:
    print(key,param[key].min(),param[key].max())
print(F.softmax(param['neck.weight']))
# param =
def get_frames(video_name,videogap):
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



def getnetinput_nat(image,center_pos,size,flagsearch):

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
    cfg.TRACK.PENALTY_K =0.1
    cfg.TRACK.WINDOW_INFLUENCE = 0.3
    cfg.TRACK.LR =0.65

    # cfg.TRAIN.SEARCH_SIZE = 160
    # cfg.TRAIN.OUTPUT_SIZE = 11
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder(cfg)
    if cfg.BACKBONE.TYPE in ["mobileones16outtwo", "mobileones8s16outtwo"]:
        if cfg.TRAIN.MODE_REPARAMETERIZE == 0:
            model.backbone = reparameterize_model_allskipscale(model.backbone)
        else:
            model.backbone = reparameterize_model_train(model.backbone)
    else:
        model.backbone = reparameterize_model(model.backbone)
    # load model
    param = torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu())
    if "state_dict" in param:
        param = param['state_dict']
    model.load_state_dict(param)
    model.eval().to(device)

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
    cnt = 0
    for frame in get_frames(args.video_name,videogap):
        cnt += 1
        print(cnt)
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                pass
            if sum(init_rect)!=0:
                print("init_rect:",init_rect)
                # init_rect = [0,0,frame.shape[1],frame.shape[0]]
                center_pos = np.array([init_rect[0] + (init_rect[2] - 1) / 2,
                                            init_rect[1] + (init_rect[3] - 1) / 2])
                size = np.array([init_rect[2], init_rect[3]])
                if cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX==4:
                    z_crop, _, _, _,_ = getnetinput_nat(frame, center_pos, size, flagsearch=False)
                else:
                    z_crop, _, _, _,_ = getnetinput(frame, center_pos, size, flagsearch=False)
                print(z_crop.size(),":zcrop")
                model.template(z_crop)
                first_frame = False
        else:
            if cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 4:

                x_crop, scalex, scaley, imgshow,croproi = getnetinput_nat(frame, center_pos, size, flagsearch=True)
            else:
                x_crop, scalex, scaley, imgshow,croproi = getnetinput(frame, center_pos, size, flagsearch=True)
            cropleft, croptop = croproi[:2]
            loc = model.track(x_crop)
            model.template(x_crop)

            cv2.namedWindow("imgshow", cv2.NORM_MINMAX)
            cv2.imshow("imgshow", imgshow)
            loc = loc.data.cpu().numpy().squeeze()
            prior_width = 1.0
            prior_height = 1.0
            prior_variance = [0.1, 0.1, 0.2, 0.2]
            prior_center_x = 0.5
            prior_center_y = 0.5
            bbox_center_x = loc[0] * prior_variance[0] * prior_width + prior_center_x
            bbox_center_y = loc[1] * prior_variance[1] * prior_height + prior_center_y
            bbox_width = math.exp(loc[2] * prior_variance[2]) * prior_width
            bbox_height = math.exp(loc[3] * prior_variance[3]) * prior_height

            bbox = [bbox_center_x * cfg.TRAIN.SEARCH_SIZE, bbox_center_y * cfg.TRAIN.SEARCH_SIZE,
                    bbox_width * cfg.TRAIN.SEARCH_SIZE, bbox_height * cfg.TRAIN.SEARCH_SIZE]

            bbox[0] /= scalex
            bbox[2] /= scalex
            bbox[1] /= scaley
            bbox[3] /= scaley

            cx = cropleft + bbox[0]
            cy = croptop + bbox[1]

            # cx = bbox[0] + center_pos[0] - size[0] * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR // 2
            # cy = bbox[1] + center_pos[1] - size[1] * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR // 2
            # cx = bbox[0] + center_pos[0]
            # cy = bbox[1] + center_pos[1]
            lr = 1.0
            # smooth bbox
            width = size[0] * (1 - lr) + bbox[2] * lr
            height = size[1] * (1 - lr) + bbox[3] * lr

            # clip boundary
            cx, cy, width, height = _bbox_clip(cx, cy, width,
                                               height, frame.shape[:2])

            # udpate state
            center_pos = np.array([cx, cy])
            size = np.array([width, height])

            bbox = [cx - width / 2,
                    cy - height / 2,
                    width,
                    height]

            # z_crop, _, _, _ = getnetinput(frame, center_pos, size, flagsearch=False)
            # model.template(z_crop)

            bbox = list(map(int, bbox))
            cx = int(bbox[0] + bbox[2] / 2)
            cy = int(bbox[1] + bbox[3] / 2)
            cv2.putText(frame, "%f" % 1.0, (int(cx), int(cy)), cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                        color=(0, 0, 255), thickness=1)
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
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
