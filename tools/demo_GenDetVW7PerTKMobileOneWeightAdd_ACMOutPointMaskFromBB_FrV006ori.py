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

from pysot.core.config import cfg
from pysot.models.wdymodel_GenDetVW7PerTKMobileOneWeightAdd_ACMOutPointMask_AnchorTarget_FrV006 import ModelBuilder
from pysot.models.backbone.mobileone_stride import reparameterize_model
from pysot.models.backbone.mobileone_strideS16OutTwo import reparameterize_model_train,reparameterize_model_allskipscale,reparameterize_models_all
from tools.bn_fusion import fuse_bn_recursively
import torch.nn.functional as F
import numpy as np
import math
from pysot.utils.bbox import corner2center, Center, center2corner, Corner
from pysot.utils.anchor import Anchors
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()



# video_name = "/home/wudengyang/track_videos/videos/PerTK2021116_Occlusion.avi"
# video_name = "/home/wudengyang/track_videos/videos/tutu.mp4"
# video_name = "/home/wudengyang/track_videos/videos/FigureSkating_copy.mp4"
# video_name = "/home/wudengyang/track_videos/videos/user_1.mp4"
# video_name = "/home/wudengyang/track_videos/videos/user_2.mp4"
# video_name = "/home/wudengyang/track_videos/videos/d8.mp4"
# video_name = "/home/wudengyang/track_videos/videos/CLIPS_3.mp4"
# video_name = "/home/wudengyang/track_videos/videos/20221114CSTHUMANS16-TEST-1CAMERA.avi"
# video_name = "/home/wudengyang/track_videos/videos/20230106CST_HUMAN_ORIG_FAIL_start29.mp4"
# video_name = "/home/wudengyang/track_videos/videos/CLIPS_3_1280x720.mp4"
video_name = "/home/wudengyang/track_videos/videos/NORM0018.MP4"
# video_name = "/home/wudengyang/track_videos/videos/NORM0062.MP4"


# args.config = "/home/wudengyang/PycharmProjects/pysot_REMO_debug/experiments/siamrpn_pertk_darknet/wdy20230218GenDetVW7PerTKMobileOneWeightAdd_ACMOutPointMask_DWV11ABShiftREMO2_4GPU125.yaml"
# args.snapshot = '/home/wudengyang/0016_GOT/wdy20230218GenDetVW7PerTKMobileOneWeightAdd_ACMOutPointMask_DWV11ABShiftREMO2_4GPU125/checkpoint_e12_param.pth'


# args.config = "/home/wudengyang/PycharmProjects/pysot_REMO_debug/experiments/siamrpn_pertk_darknet/wdy20230223GenDetVW7PerTKMobileOneWeightAdd_ACMOutPointMask_DWV11ABShiftREMO2_4GPU125.yaml"
# args.snapshot = '/home/wudengyang/0016_GOT/wdy20230223GenDetVW7PerTKMobileOneWeightAdd_ACMOutPointMask_DWV11ABShiftREMO2_4GPU125/checkpoint_e1_param.pth'

# args.config = "/home/wudengyang/PycharmProjects/pysot_REMO_debug/experiments/siamrpn_pertk_darknet/wdy20230223GenDetVW7PerTKMobileOneWeightAdd_ACMOutPointMask_DWV11ABShiftREMO2_4GPU125.yaml"
# args.snapshot = '/home/wudengyang/0016_GOT/wdy20230223GenDetVW7_ABScale3Ratio2big_DWV11ABShiftREMO2batch8_4GPU128/checkpoint_e2_param.pth'
#
# args.config = "/home/wudengyang/PycharmProjects/pysot_REMO_debug/experiments/siamrpn_pertk_darknet/wdy20230223GenDetVW7_ABScale3Ratio2_DWV11ABShiftREMO4batch8_4GPU128.yaml"
# args.snapshot = '/home/wudengyang/0016_GOT/wdy20230223GenDetVW7_ABScale3Ratio2_DWV11ABShiftREMO4batch8_4GPU128/checkpoint_e1_param.pth'

# args.config = "/home/wudengyang/PycharmProjects/pysot_REMO_debug/experiments/siamrpn_pertk_darknet/wdy20230223GenDetVW7_ABScale7_10.5_14Ratio0.9_1.3_DWV11ABShiftREMO2batch8_4GPU128.yaml"
# args.snapshot = '/home/wudengyang/0016_GOT/wdy20230223GenDetVW7_ABScale7_10.5_14Ratio0.9_1.3_DWV11ABShiftREMO2batch8_4GPU128/checkpoint_e16.pth'

# args.config = "/home/wudengyang/PycharmProjects/pysot_REMO_debug/experiments/siamrpn_pertk_darknet/wdy20230224GenDetVW7_ABScale8_11_15Ratio0.95_1.3_DWV11ABShiftREMO2batch8_4GPU125.yaml"
# args.snapshot = '/home/wudengyang/0016_GOT/wdy20230224GenDetVW7_ABScale8_11_15Ratio0.95_1.3_DWV11ABShiftREMO2batch8_4GPU125/checkpoint_e19_param.pth'


args.config = "/home/wudengyang/PycharmProjects/pysot_REMO_debug/experiments/siamrpn_pertk_darknet/wdy20230225GenDetVW7_ABScale7_10.5_15Ratio0.9_1.3_DWV14ABShiftREMO2batch8_4GPU125.yaml"
args.snapshot = '/home/wudengyang/0016_GOT/wdy20230225GenDetVW7_ABScale7_10.5_15Ratio0.9_1.3_DWV14ABShiftREMO2batch8_4GPU125/checkpoint_e19_param.pth'


# args.config = "/home/wudengyang/PycharmProjects/pysot_REMO_debug/experiments/siamrpn_pertk_darknet/wdy20230224GenDetVW7_ABScale9.5_13_15Ratio0.85_1.3_DWV11ABShiftREMO2batch8_4GPU128.yaml"
# args.snapshot = '/home/wudengyang/0016_GOT/wdy20230224GenDetVW7_ABScale9.5_13_15Ratio0.85_1.3_DWV11ABShiftREMO2batch8_4GPU128/checkpoint_e15_param.pth'

# args.config = "/home/wudengyang/PycharmProjects/pysot_REMO_debug/experiments/siamrpn_pertk_darknet/wdy20230227GenDetVW7_ABScale7_10.5_14Ratio0.9_1.3_DWV14ABtarget1ShiftREMO2batch8_4GPU128.yaml"
# args.snapshot = '/home/wudengyang/0016_GOT/wdy20230227GenDetVW7_ABScale7_10.5_14Ratio0.9_1.3_DWV14ABtarget1ShiftREMO2batch8_4GPU128/checkpoint_e19_param.pth'





# video_name = 0
if video_name == 0:
    nwait = 1
else:
    nwait = 0
args.video_name = video_name
param =torch.load(args.snapshot,map_location=torch.device('cpu'))
for key in param:
    if 'rpn_head' in key:
        print(key,param[key].min(),param[key].max())
# param =
def get_frames(video_name):
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


def getnetinput(image, center_pos, size, flagsearch=False):

    im_h, im_w = image.shape[:2]
    cx, cy = center_pos
    bw, bh = size
    x1 = cx-bw/2
    x2 = x1 + bw
    y1 = cy - bh/2
    y2 = y1 + bh
    cropw = bw*cfg.DATASET.PERSONTK.KCONTEXTFACTOR
    croph = bh*cfg.DATASET.PERSONTK.KCONTEXTFACTOR
    if cfg.DATASET.PERSONTK.FLAG_TEMPLATESEARCH_SIZEFROMCONFIG:
        if flagsearch:
            scale = float(cfg.TRAIN.SEARCH_SIZE) / float(cfg.TRAIN.EXEMPLAR_SIZE)
            cropw *= scale
            croph *= scale
    print(cropw,bw,"crop",flagsearch,cfg.DATASET.PERSONTK.FLAG_TEMPLATESEARCH_SIZEFROMCONFIG)
    xmin = int(cx - cropw / 2)
    xmax = int(cx + cropw / 2)
    ymin = int(cy - croph / 2)
    ymax = int(cy + croph / 2)
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
    h1, w1 = img_pad.shape[:2]
    if cfg.DATASET.PERSONTK.FLAG_TEMPLATESEARCH_SIZEFROMCONFIG:
        if flagsearch:
            im_patch = cv2.resize(img_pad, (cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE))
            sx = float(cfg.TRAIN.SEARCH_SIZE) / w1
            sy = float(cfg.TRAIN.SEARCH_SIZE) / h1
        else:
            im_patch = cv2.resize(img_pad, (cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE))
            sx = float(cfg.TRAIN.EXEMPLAR_SIZE) / w1
            sy = float(cfg.TRAIN.EXEMPLAR_SIZE) / h1
    else:
        im_patch = cv2.resize(img_pad, (cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE))
        sx = float(cfg.TRAIN.SEARCH_SIZE)/w1
        sy = float(cfg.TRAIN.SEARCH_SIZE)/h1
    imgshow = im_patch.copy()
    if flagsearch:
        cv2.namedWindow("imgsearch", cv2.NORM_MINMAX)
        cv2.imshow("imgsearch", imgshow)
    else:
        cv2.namedWindow("imgt", cv2.NORM_MINMAX)
        cv2.imshow("imgt", imgshow)
    im_patch = im_patch.transpose(2, 0, 1)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)
    im_patch = torch.from_numpy(im_patch)
    if cfg.CUDA:
        im_patch = im_patch.cuda()
    return im_patch, sx, sy, imgshow



def generate_anchor(score_size):
    anchors = Anchors(cfg.ANCHOR.STRIDE,
                      cfg.ANCHOR.RATIOS,
                      cfg.ANCHOR.SCALES)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
    total_stride = anchors.stride
    anchor_num = anchor.shape[0]
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)], [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor

def _convert_bbox(delta, point):
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
    delta = delta.detach().cpu().numpy()

    delta[0, :] = point[:, 0] - delta[0, :]
    delta[1, :] = point[:, 1] - delta[1, :]
    delta[2, :] = point[:, 0] + delta[2, :]
    delta[3, :] = point[:, 1] + delta[3, :]
    delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
    return delta

def _convert_bbox1(delta, anchor):
    delta = delta.reshape(1, 4, -1, 11, 11).permute(1, 2, 3, 4, 0).contiguous().view(4, -1)
    delta = delta.detach().cpu().numpy()

    delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
    delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
    return delta
def _convert_score(score):
    # if self.cls_out_channels == 1:
    #     score = score.permute(1, 2, 3, 0).contiguous().view(-1)
    #     score = score.sigmoid().detach().cpu().numpy()
    # else:
    score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
    if cfg.TRAIN.FLAG_SIGMOID_LOSS:
        # score = score.sigmoid()
        score = score.detach()[:, 1].cpu().numpy()
    else:
        score = score.softmax(1).detach()[:, 1].cpu().numpy()
    return score
def _convert_score1(score):
    # if self.cls_out_channels == 1:
    #     score = score.permute(1, 2, 3, 0).contiguous().view(-1)
    #     score = score.sigmoid().detach().cpu().numpy()
    # else:
    score = score.reshape(1, 2, -1, 11, 11).permute(1, 2, 3, 4, 0).contiguous().view(2, -1).permute(1, 0)
    if cfg.TRAIN.FLAG_SIGMOID_LOSS:
        # score = score.sigmoid()
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

    # cfg.TRACK.PENALTY_K =0.0
    # cfg.TRACK.WINDOW_INFLUENCE = 0.0
    # cfg.TRACK.LR =1.0

    cfg.TRACK.PENALTY_K = 0.017
    cfg.TRACK.WINDOW_INFLUENCE = 0.3
    cfg.TRACK.LR = 0.85

    # cfg.TRACK.PENALTY_K = 0.1
    # cfg.TRACK.WINDOW_INFLUENCE = 0.4
    # cfg.TRACK.LR = 0.85

    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()
    if cfg.BACKBONE.TYPE == "mobileones16outtwo":
        if cfg.TRAIN.MODE_REPARAMETERIZE == 0:
            model.backbone = reparameterize_model_train(model.backbone)
        else:
            model.backbone = reparameterize_model_allskipscale(model.backbone)
    else:

        model.backbone = reparameterize_model(model.backbone)

    if "model.pth" in args.snapshot:
        model.load_state_dict(torch.load(args.snapshot,
                                         map_location=lambda storage, loc: storage.cpu()))
    else:
        state_dict = model.state_dict()
        param = torch.load(args.snapshot, map_location=lambda storage, loc: storage.cpu())
        if "state_dict" in param:
            param = param["state_dict"]
        keylistsrc = list(param.keys())
        keylistdst = list(state_dict.keys())
        n = min(len(keylistsrc), len(keylistdst))
        # print(keylistsrc)
        # print(keylistdst)
        # for i in range(n):
        #     print(i, keylistsrc[i], keylistdst[i])
        for key in state_dict:
            state_dict[key] = param[key]
        model.load_state_dict(param, strict=True)

    # load model
    # model.load_state_dict(torch.load(args.snapshot,
    #     map_location=lambda storage, loc: storage.cpu()))
    model = reparameterize_models_all(model)
    model = fuse_bn_recursively(model)
    model = model.eval().to(device)

    # build tracker
    anchors = generate_anchor(cfg.TRAIN.OUTPUT_SIZE)
    hanning = np.hanning(cfg.TRAIN.OUTPUT_SIZE)
    window = np.outer(hanning, hanning)
    window = np.tile(window.flatten(), cfg.ANCHOR.ANCHOR_NUM)
    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'

    root_path = '/home/wudengyang/track_videos/results'
    test_video_name = os.path.basename(args.video_name)
    model_name = os.path.split(args.config)[1][:-5]
    save_model_path = os.path.join(root_path, model_name)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    epoch_num = os.path.basename(args.snapshot).split('.')[0].split('_')[1]
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cap = cv2.VideoCapture(args.video_name)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out = cv2.VideoWriter(f'{root_path}/{model_name}/{epoch_num}_{test_video_name}', fourcc, 10, (int(width), int(height)))
    cur_frames = 0
    nframe = 3
    for frame in get_frames(args.video_name):
        cur_frames += 1
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                pass
            # init_rect = (639, 89, 332, 462)
            if sum(init_rect) != 0:
                print("init_rect:", init_rect)
                # init_rect = [0,0,frame.shape[1],frame.shape[0]]
                center_pos = np.array([init_rect[0] + (init_rect[2] - 1) / 2,
                                       init_rect[1] + (init_rect[3] - 1) / 2])
                size = np.array([init_rect[2], init_rect[3]])
                z_crop, _, _, _ = getnetinput(frame, center_pos, size, flagsearch=False)
                model.template(z_crop)
                first_frame = False
        else:
            if cur_frames % nframe == 0:
                x_crop, scalex, scaley, imgshow = getnetinput(frame, center_pos, size, flagsearch=True)
            else:
                continue
            cls, loc, maskpred = model.track(x_crop)
            cls = cls.sigmoid()


            for i in range(cfg.ANCHOR.ANCHOR_NUM, cfg.ANCHOR.ANCHOR_NUM * 2):
                cls_show = cls.clone()
                cls_show = cls_show[0, i].data.cpu().numpy()
                cls_showorg = cls_show.copy()
                cls_showorg *= 255
                cls_showorg = cls_showorg.astype(np.uint8)
                cv2.namedWindow(f"cls_showorg{i}", cv2.NORM_MINMAX)
                cv2.imshow(f"cls_showorg{i}", cls_showorg)
                head1 = cv2.resize(cls_show, (cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE))
                head1 *= 255
                head1 = head1.astype(np.uint8)
                heat_img = cv2.applyColorMap(head1, cv2.COLORMAP_JET)
                add_img = cv2.addWeighted(imgshow, 0.5, heat_img, 0.5, 0)
                cv2.namedWindow(f"cls_show{i}", cv2.NORM_MINMAX)
                cv2.imshow(f"cls_show{i}", add_img)


            score = _convert_score1(cls)
            pred_bbox = _convert_bbox1(loc, anchors)
            scoretmp = score.copy()
            idx = scoretmp.argsort()[::-1]
            # pred_bbox_good = pred_bbox[:, idx[:5]]


            def change(r):
                return np.maximum(r, 1. / r)

            t1 = pred_bbox[2, :]
            t2 = pred_bbox[3, :]
            print(f't1.min(), t2.min():{t1.min(), t2.min()}')
            # scale penalty
            szpred = np.sqrt(pred_bbox[2, :]*pred_bbox[3, :])
            s_c = change(szpred/80.0)

            # aspect ratio penalty
            r_c = change(1.0 / (pred_bbox[2, :] / pred_bbox[3, :]))
            penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
            pscore = penalty * score

            # window penalty
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     window * cfg.TRACK.WINDOW_INFLUENCE
            print(f'score.max(), pscore.max():{score.max(), pscore.max()}')

            sort_pscore = pscore.argsort()[::-1]
            best_idx = sort_pscore[0]


            bbox = pred_bbox[:, best_idx]
            bbox[0] /= scalex
            bbox[2] /= scalex
            bbox[1] /= scaley
            bbox[3] /= scaley
            print(f'bbox:{bbox}')



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

            # pred_bbox_good[0] /= scalex
            # pred_bbox_good[2] /= scalex
            # pred_bbox_good[1] /= scaley
            # pred_bbox_good[3] /= scaley
            #
            # pred_bbox_good[0, :] += center_pos[0]
            # pred_bbox_good[1, :] += center_pos[1]
            # udpate state
            center_pos = np.array([cx, cy])
            size = np.array([width, height])

            bbox = [cx - width / 2,
                    cy - height / 2,
                    width,
                    height]
            best_score = score[best_idx]


            # z_crop, _, _, _ = getnetinput(frame, center_pos, size, flagsearch=False)
            # model.template(z_crop)

            bbox = list(map(int, bbox))
            cx = int(bbox[0]+bbox[2]/2)
            cy = int(bbox[1]+bbox[3]/2)
            cv2.putText(frame, "%f" % best_score, (int(cx), int(cy)), cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                        color=(0, 0, 255), thickness=1)
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 0, 255), 3)


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


            out.write(frame)
            cv2.imshow(video_name, frame)
            key = cv2.waitKey(nwait)
            if key == 27:
                exit()
            # cv2.waitKey(1)
    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()



