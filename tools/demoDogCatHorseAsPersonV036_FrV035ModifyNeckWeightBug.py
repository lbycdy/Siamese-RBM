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
from siamban.models.model_DogCatHorseAsPersonV041_FrV039ConcatSearchFeatAffCorr import ModelBuilder
from siamban.models.backbone.mobileone_stride import reparameterize_model
from siamban.models.backbone.mobileone_strideS16OutTwo import reparameterize_model_allskipscale,reparameterize_model_train,reparameterize_models_all
import torch.nn.functional as F
import numpy as np
import math
from bn_fusion import fuse_bn_recursively
from siamban.utils.bbox import corner2center, \
        Center, center2corner, Corner
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()
def calciou(box1, box2):
    x1_1 = box1[0]
    y1_1 = box1[1]
    x2_1 = box1[2]
    y2_1 = box1[3]
    x1_2 = box2[0]
    y1_2 = box2[1]
    x2_2 = box2[2]
    y2_2 = box2[3]
    interx = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    intery = max((0, min(y2_1, y2_2) - max(y1_1, y1_2)))
    area_inter = interx * intery
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    return float(area_inter) / float(area1 + area2 - area_inter + 1e-8)


def nms(score_all,boxes4p_all):
    voting_thre = 0.85
    nms_thre = 0.35
    voting_info = {}
    idxsort = np.argsort(-score_all)
    for idx in idxsort:
        keep = True
        for idxkeep in voting_info:
            overlap = calciou(boxes4p_all[idxkeep], boxes4p_all[idx])
            if overlap > voting_thre:
                voting_info[idxkeep].append([score_all[idx], boxes4p_all[idx]])
            if overlap > nms_thre:
                keep = False
        if keep:
            voting_info[idx] = [[score_all[idx], boxes4p_all[idx]], ]
    boxnms4p = []
    for idx in voting_info:
        s_all = 0
        for s, box in voting_info[idx]:
            s_all += s
        boxvot = np.zeros((4,))
        for s, box in voting_info[idx]:
            scale = s / s_all
            boxvot += box * scale
        boxnms4p.append([score_all[idx], boxvot])
    return boxnms4p






args.config = "/home/lbycdy/work/siamban/experiments/siamban_r50_l234/lb20230612DogCatHorseAsPersonV041_FrV039ConcatSearchFeatAffCorrDV19MultGTAsDetCropTCat0.5Other0.1S0.6MaxS0.75SFT1PS3.0covea0.3_4GPU127.yaml"
args.snapshot = '/home/lbycdy/checkpoint/lb20230612DogCatHorseAsPersonV041_FrV039ConcatSearchFeatAffCorrDV19MultGTAsDetCropTCat0.5Other0.1S0.6MaxS0.75SFT1PS3.0covea0.3_4GPU127/checkpoint_e1_param.pth'

# args.config = "/home


video_name = "/home/lbycdy/Videos/vlc-record-2023-01-09-16h44m36s-Cat10.mp4-.mp4"
video_name = "/home/lbycdy/Videos/Cat10_start2070.mp4"
# video_name = "/home/lbycdy/Videos/Cat10_start2070_Frm0To1rep500.mp4"
# video_name = "/home/lbycdy/Videos/Cat10_start2070_Frm0To50LastRep400.mp4"
# video_name = "/home/lbycdy/datasets/cow-got/cow_video/ch05_20201003111111.mp4"
# video_name = "/home/lbycdy/Videos/1.mp4"

# video_name = "/home/lbycdy/Videos/cat10.mp4"
# video_name = "/home/lbycdy/datasets/DogCatHorseVideo/Video1103/Cat/Cat10.mp4"
# video_name = "/home/lbycdy/datasets/DogCatHorseVideo/Video1103/Cat/MyVideo_1.mp4"
# video_name = "/home/lbycdy/datasets/DogCatHorseVideo/Video1103/Horse/Horse07.mp4"
# video_name = "/home/lbycdy/datasets/DogCatHorseVideo/Video1103/Dog/Dog02.mp4"
videogap = 3
flagupdatetemplate = False
args.video_name = video_name
param =torch.load(args.snapshot,map_location=torch.device('cpu'))
if "state_dict" in param:
    param = param['state_dict']
for key in param:
    print(key,param[key].min(),param[key].max())
try:
    print(F.softmax(param['neck.weight']),param['neck.weight'])
except:
    pass

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
    if cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX==0:
        if cfg.DATASET.CATDOGHORSETK.SHIFT_MODE==8 and flagsearch:
            cropw = bw + 0.5*(bw+bh)
            croph = bh + 0.5*(bw+bh)

        else:
            cropw = bw * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
            croph = bh * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
  

    elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX==1:
        cropw = bw + 0.5 * (bw + bh)
        croph = bh + 0.5 * (bw + bh)
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
    cropleft,croptop = xmin,ymin
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
    return im_patch,sx,sy,imgshow,cropleft,croptop



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
    return im_patch,sx,sy,imgshow

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
    deltacenter = delta.copy()
    deltacenter[0, :], deltacenter[1, :], deltacenter[2, :], deltacenter[3, :] = corner2center(deltacenter)
    return delta,deltacenter

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
    cfg.TRACK.PENALTY_K = 0.1
    cfg.TRACK.WINDOW_INFLUENCE = 0.3
    cfg.TRACK.LR = 0.65

    # cfg.TRAIN.SEARCH_SIZE = 160
    # cfg.TRAIN.OUTPUT_SIZE = 11

    # cfg.DATASET.CATDOGHORSETK.FLAG_TEMPLATESEARCH_SIZEFROMCONFIG = True
    # cfg.TRAIN.SEARCH_SIZE = 224
    # cfg.TRAIN.OUTPUT_SIZE = 27
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder(cfg)
    model.backbone = reparameterize_model_allskipscale(model.backbone)
    # load model
    param = torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu())
    if "state_dict" in param:
        param = param['state_dict']
    model.load_state_dict(param)
    model = reparameterize_models_all(model)
    model = fuse_bn_recursively(model)

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
    colorlist = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for frame in get_frames(args.video_name,videogap):
        cnt += 1
        print(cnt)
        if first_frame:
            init_rect = (287, 212, 761, 328)
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

                center_pos1 = np.array([init_rect[0]-50,
                                       init_rect[1] + (init_rect[3] - 1) / 2])
                size1 = np.array([init_rect[2]*4, init_rect[3]*2])
                z_crop, _, _, _,_,_ = getnetinput(frame, center_pos, size, flagsearch=False)
                print(z_crop.size(),":zcrop")
                model.template(z_crop)
                first_frame = False
                # center_pos[0]+=200
                # center_pos[1]+=50

        else:
            x_crop, scalex, scaley, imgshow,cropleft,croptop = getnetinput(frame, center_pos, size, flagsearch=True)
            print(x_crop.size(),":x_crop")
            cls,loc = model.track(x_crop)
            print("cls:",cls.sigmoid().min(),cls.sigmoid().max(),cls.sigmoid().mean())
            print("loc:",loc.min(),loc.max(),loc.mean())

            if flagupdatetemplate:
                model.template(x_crop)
            cls_show = cls.clone()
            if cfg.TRAIN.FLAG_SIGMOID_LOSS:
                cls_show = cls_show.sigmoid()[0,1].data.cpu().numpy()

            else:
                cls_show = cls_show.softmax(1)[0,1].data.cpu().numpy()
            cls_showorg = cls_show.copy()
            cls_showorg *= 255
            cls_showorg = cls_showorg.astype(np.uint8)
            cv2.namedWindow("cls_showorg", cv2.NORM_MINMAX)
            cv2.imshow("cls_showorg", cls_showorg)
            head1 = cv2.resize(cls_show, (cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE))
            head1 *= 255
            head1 = head1.astype(np.uint8)
            heat_img = cv2.applyColorMap(head1, cv2.COLORMAP_JET)
            add_img = cv2.addWeighted(imgshow, 0.5, heat_img, 0.5, 0)
            cv2.namedWindow("cls_show", cv2.NORM_MINMAX)
            cv2.imshow("cls_show", add_img)
            score = _convert_score(cls)
            pred_bbox_corner,pred_bbox = _convert_bbox(loc, points)
            scoretmp = score.copy()
            idx = scoretmp.argsort()[::-1]
            print(pred_bbox.shape)
            pred_bbox_good = pred_bbox[:,idx[:40]]
            print(score.shape,pred_bbox.shape)
            flag = score>0.8
            score_high  = score[flag]
            pred_bbox_high = pred_bbox_corner[:,flag]+80
            print(score_high.shape,pred_bbox_high.shape)
            boxnms = nms(score_high,pred_bbox_high.transpose((1,0)))
            print(len(boxnms),":______________________",pred_bbox_high.shape)
            print(pred_bbox_corner.shape,points.shape,imgshow.shape,points.min(),points.max())
            for i in range(len(boxnms)):

                s, box = boxnms[i]
                xmin,ymin,xmax,ymax = box
                xmin,ymin,xmax,ymax = map(int,[xmin,ymin,xmax,ymax])
                cv2.rectangle(imgshow, (xmin, ymin), (xmax, ymax),
                                                colorlist[i % 5], 3)
            cv2.namedWindow("imgshow", cv2.NORM_MINMAX)
            cv2.imshow("imgshow", imgshow)
            def change(r):
                return np.maximum(r, 1. / r)
            def sz(w,h):
                return np.sqrt(w*h)
            t1 = pred_bbox[2, :]
            t2 = pred_bbox[3, :]
            print(t1.min(),t2.min())
            if cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX==0:
                # scale penalty
                szpred = np.sqrt(pred_bbox[2, :]*pred_bbox[3, :])
                s_c = change(szpred/80.0)

                # aspect ratio penalty
                r_c = change(1.0 /
                             (pred_bbox[2, :] / pred_bbox[3, :]))
            else:
                szpred = np.sqrt(pred_bbox[2, :] * pred_bbox[3, :])
                s1 = np.sqrt(size[0]*scalex*size[1]*scaley)
                s_c = change(szpred / s1)
                r1 = size[0]*scalex/(size[1]*scaley)
                r_c = change(r1 /
                             (pred_bbox[2, :] / pred_bbox[3, :]))
                print("s1:r1---",s1,r1)
            penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)

            pscore = penalty * score
            print(score.max(),pscore.max())

            # window penalty
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     window * cfg.TRACK.WINDOW_INFLUENCE
            best_idx = np.argmax(pscore)
            bbox = pred_bbox[:, best_idx]
            print("best_idx:",best_idx)
            bbox[0] /= scalex
            bbox[2] /= scalex
            bbox[1] /= scaley
            bbox[3] /= scaley

            s = penalty[best_idx] * score[best_idx]
            lr = s * cfg.TRACK.LR

            cx = bbox[0] + center_pos[0]
            cy = bbox[1] + center_pos[1]

            # smooth bbox
            width = size[0] * (1 - lr) + bbox[2] * lr
            height = size[1] * (1 - lr) + bbox[3] * lr

            # clip boundary
            cx, cy, width, height = _bbox_clip(cx, cy, width,
                                                    height, frame.shape[:2])

            pred_bbox_good[0] /= scalex
            pred_bbox_good[2] /= scalex
            pred_bbox_good[1] /= scaley
            pred_bbox_good[3] /= scaley

            pred_bbox_good[0, :] += center_pos[0]
            pred_bbox_good[1, :] += center_pos[1]

            for i in range(len(boxnms)):
                s, box = boxnms[i]
                box[0] /= scalex
                box[2] /= scalex
                box[1] /= scaley
                box[3] /= scaley
                xmin = box[0] + cropleft
                ymin = box[1] + croptop
                xmax = box[2] + cropleft
                ymax = box[3] + croptop
                xmin,ymin,xmax,ymax = map(int,[xmin,ymin,xmax,ymax])
                # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                #                                 colorlist[i % 5], 3)
            # udpate state
            center_pos = np.array([cx, cy])
            size = np.array([width, height])

            bbox = [cx - width / 2,
                    cy - height / 2,
                    width,
                    height]
            best_score = score[best_idx]


            bbox = list(map(int, bbox))
            print("bbox:", bbox)
            cx = int(bbox[0]+bbox[2]/2)
            cy = int(bbox[1]+bbox[3]/2)
            cv2.putText(frame, "%f" % best_score, (int(cx), int(cy)), cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                        color=(0, 0, 255), thickness=1)
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 0, 255), 3)
                # tracker.init(frame, bbox)

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
            if key==27:
                exit()


if __name__ == '__main__':
    main()
