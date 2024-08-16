# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os


import cv2
import numpy as np
from torch.utils.data import Dataset

from siamban.utils.bbox import center2corner, Center
from siamban.core.config import cfg
import random
import math
from siamban.utils.bbox import corner2center, \
        Center, center2corner, Corner

import socket
import os.path as osp
import _pickle as pickle

hostname = socket.gethostname()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
ipstr = s.getsockname()[0]
ipaddress_int = int(ipstr.split('.')[-1])

logger = logging.getLogger("global")


# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


def load_pickle(path):
  """Check and load pickle object.
  According to this post: https://stackoverflow.com/a/41733927, cPickle and
  disabling garbage collector helps with loading speed."""
  # print(path)
  assert osp.exists(path)
  # gc.disable()
  with open(path, 'rb') as f:
    f.seek(0)
    ret = pickle.load(f)
  # gc.enable()
  return ret
def shift_random():
    return np.random.random() * 2 - 1.0

def get_det_subwindow(im, pos, bboxes, model_sz, original_sz, avg_chans):
    """
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
        avg_chans: channel average
    """
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    # context_xmin = round(pos[0] - c) # py2 and py3 round
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    # context_ymin = round(pos[1] - c)
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    shift = 128
    sx = shift_random() * shift
    sy = shift_random() * shift
    x1, y1, x2, y2 = context_xmin, context_ymin, context_xmax + 1, context_ymax + 1

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans

        sx = max(-x1, min(c + left_pad + right_pad - 1 - x2, sx))
        sy = max(-y1, min(r + top_pad + bottom_pad - 1 - y2, sy))
        crop_box = Corner(x1+sx, y1+sy, x2+sx, y2+sy)
        im_patch = te_im[int(y1 + sy):int(y2 + sy),
                   int(x1 + sx):int(x2 + sx), :]
        boxes = []
        for box in bboxes:
            b_x1, b_y1, b_x2, b_y2 = box[0]+left_pad, box[1]+top_pad, box[2]+left_pad, box[3]+top_pad
            shift_box = [b_x1-crop_box.x1, b_y1-crop_box.y1, b_x2-crop_box.x1, b_y2-crop_box.y1]
            i_w = max(0, min(shift_box[2], original_sz-1) - max(shift_box[0], 0))
            i_h = max(0, min(shift_box[3], original_sz-1) - max(shift_box[1], 0))
            interarea = i_w * i_h
            bbox_square = (shift_box[2] - shift_box[0]) * (shift_box[3] - shift_box[1])
            interarea_ratio = interarea / bbox_square
            if interarea_ratio > 0.5:
                boxes.append([max(shift_box[0], 0), max(shift_box[1], 0), min(shift_box[2], original_sz-1), min(shift_box[3], original_sz-1), box[4]])


    else:
        sx = max(-x1, min(im_sz[1] - 1 - x2, sx))
        sy = max(-y1, min(im_sz[0] - 1 - y2, sy))
        crop_box = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)
        im_patch = im[int(y1 + sy):int(y2 + sy),
                   int(x1 + sx):int(x2 + sx), :]

        boxes = []
        for box in bboxes:
            b_x1, b_y1, b_x2, b_y2 = box[0], box[1], box[2], box[3]
            shift_box = [b_x1-crop_box.x1, b_y1-crop_box.y1, b_x2-crop_box.x1, b_y2-crop_box.y1]
            i_w = max(0, min(shift_box[2], original_sz-1) - max(shift_box[0], 0))
            i_h = max(0, min(shift_box[3], original_sz-1) - max(shift_box[1], 0))
            interarea = i_w * i_h
            bbox_square = (shift_box[2] - shift_box[0]) * (shift_box[3] - shift_box[1])
            interarea_ratio = interarea / bbox_square
            if interarea_ratio > 0.5:
                boxes.append([max(shift_box[0], 0), max(shift_box[1], 0), min(shift_box[2], original_sz-1), min(shift_box[3], original_sz-1), box[4]])


    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        box_crop = []
        for b in boxes:
            scale_x = model_sz/original_sz
            box_scale = [b[i]*scale_x for i in range(len(b)-1)]
            box_scale = [max(box_scale[0], 0), max(box_scale[1], 0), min(box_scale[2], model_sz-1), min(box_scale[3], model_sz-1), b[4]]
            # box_scale.append(b[4])
            box_crop.append(box_scale)
        return im_patch, box_crop
    else:
        return im_patch, boxes


def crop_detection_img(image, bboxes):
    center_box = bboxes[0]
    size = np.array([center_box[2]-center_box[0], center_box[3]-center_box[1]])
    center_pos = np.array([(center_box[2] + center_box[0]) / 2., (center_box[3] + center_box[1]) / 2.])
    w_z = size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
    h_z = size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
    s_z = np.sqrt(w_z * h_z)
    s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
    channel_average = np.mean(image, axis=(0, 1))
    d_crop = None
    bboxes_crop = []
    while len(bboxes_crop) == 0:
        d_crop, bboxes_crop = get_det_subwindow(image, center_pos, bboxes,
                                        cfg.TRAIN.DETECTION_SIZE,
                                        round(s_x), channel_average)

    return d_crop, bboxes_crop



class OneImgSeg(object):
    def __init__(self, img_path, msk_root,mask_json):
        self.img_path = img_path
        self.mask_json = mask_json
        self.msk_root = msk_root
    def get_img(self):
        img = cv2.imread(self.img_path)
        return img
    def get_random_person(self,flagmask=True):
        maskname,box = random.choice(self.mask_json)
        if flagmask:
            segmentation = cv2.imread(os.path.join(self.msk_root,maskname),flags=cv2.IMREAD_GRAYSCALE)
            return segmentation,box
        else:
            return box



class Point:
    """
    This class generate points.
    """
    def __init__(self, stride, size, image_center):
        self.stride = stride
        self.size = size
        self.image_center = image_center

        self.points = self.generate_points(self.stride, self.size, self.image_center)

    def generate_points(self, stride, size, im_c):
        ori = im_c - size // 2 * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((2, size, size), dtype=np.float32)
        points[0, :, :], points[1, :, :] = x.astype(np.float32), y.astype(np.float32)

        return points


class PointTarget:
    def __init__(self, ):
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)

    def __call__(self, target, size, neg=False):

        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((size, size), dtype=np.int64)
        delta = np.zeros((4, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)
        points = self.points.points

        if neg:
            neg = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                           np.square(tcy - points[1]) / np.square(th / 4) < 1)
            neg, neg_num = select(neg, cfg.TRAIN.NEG_NUM)
            cls[neg] = 0

            return cls, delta

        delta[0] = points[0] - target[0]
        delta[1] = points[1] - target[1]
        delta[2] = target[2] - points[0]
        delta[3] = target[3] - points[1]

        # ellipse label
        pos = np.where(np.square(tcx - points[0]) / np.square(tw / cfg.POINT.SCALE_POS) +
                       np.square(tcy - points[1]) / np.square(th / cfg.POINT.SCALE_POS) < 1)
        neg = np.where(np.square(tcx - points[0]) / np.square(tw / cfg.POINT.SCALE_NEG) +
                       np.square(tcy - points[1]) / np.square(th / cfg.POINT.SCALE_NEG) > 1)

        # sampling
        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

        cls[pos] = 1
        cls[neg] = 0

        return cls, delta

class BANDataset(Dataset):
    def __init__(self,world_size=None,batchsize=None):
        super(BANDataset, self).__init__()
        self.path_format = '{}.{}.{}.jpg'

        # create point target
        self.point_target = PointTarget()
        # dicn = {0:"person",1:"cat",2:"dog",3:"horse"}
        # create sub dataset
        self.clstag2clsid_dict = {
            "n02084071": 2,  # DET and VID
            'n02121808': 1,
            'n02374451': 3,
            'n02391049': 3,  # zebra
            '17': 1,  # COCO
            '18': 2,
            '19': 3,
            '24': 3,  # zerba
            'cat': 1,  # got10k,ytbb,lasot
            'dog': 2,
            'horse': 3,
            'zebra': 3,
            'hinny': 3,
            'mule': 3,
            'canine': 2
        }
        self.datalist_img = []
        self.datalist_vid = []
        self.datalist_other = []
        self.root_list = []
        self.dataset_img = []
        self.det_bbox = {}


        rootid = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            root = subdata_cfg.ROOT
            anno = subdata_cfg.ANNO
            num_use_dog = 5
            num_use_cat = 5
            num_use_horse = 5
            num_use_other = 5

            data = json.load(open(anno))
            cat_list = []
            dog_list = []
            horse_list = []
            other_list = []
            if name == "REMOCATDOGHORSE":
                data = json.load(open(anno))
                for imgpath in data:
                    img_path = os.path.join(root, imgpath)
                    self.det_bbox[img_path] = []
                    for box in data[imgpath]:
                        xmin, ymin, xmax, ymax, cid = box
                        boxinfo = ['remo', rootid, imgpath, [xmin, ymin, xmax, ymax]]
                        if cid == 1:
                            cat_list.append(boxinfo)
                        elif cid == 2:
                            dog_list.append(boxinfo)
                        elif cid == 3:
                            horse_list.append(boxinfo)
                        self.det_bbox[img_path].append([xmin, ymin, xmax, ymax, 1])
                    self.dataset_img.append(img_path)
            cat_list = self.getnumdata(cat_list, num_use_cat)
            dog_list = self.getnumdata(dog_list, num_use_dog)
            horse_list = self.getnumdata(horse_list, num_use_horse)
            # other_list = self.getnumdata(other_list, num_use_other)
            if name in ['REMOCATDOGHORSE', 'COCO', 'DET']:
                if len(cat_list) > 0:
                    self.datalist_img.extend(cat_list)
                if len(dog_list) > 0:
                    self.datalist_img.extend(dog_list)
                if len(horse_list) > 0:
                    self.datalist_img.extend(horse_list)
            else:
                if len(cat_list) > 0:
                    self.datalist_vid.extend(cat_list)
                if len(dog_list) > 0:
                    self.datalist_vid.extend(dog_list)
                if len(horse_list) > 0:
                    self.datalist_vid.extend(horse_list)
            # if len(other_list) > 0:
            #     self.datalist_other.extend(other_list)
            info1 = "{} {} cat; {} dog; {} horse; {} other.".format(name,len(cat_list),len(dog_list),len(horse_list),len(other_list))
            info2 = "{} {} datalist_img; {} datalist_vid; {} datalist_other.".format(name,len(self.datalist_img),len(self.datalist_vid),len(self.datalist_other))
            logger.info(info1)
            logger.info(info2)
            print(info1)
            print(info2)

            root = subdata_cfg.ROOT
            self.root_list.append(root)
            rootid += 1

        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        if world_size is not None and batchsize is not None:
            if self.num % (batchsize * world_size) != 0:
                n = self.num // (batchsize * world_size)
                self.num = (n + 1) * (batchsize * world_size)
        print(self.num,":num")
        random.shuffle(self.datalist_img)
        random.shuffle(self.datalist_vid)
        random.shuffle(self.datalist_other)
        random.shuffle(self.dataset_img)
        self.num_det_img = len(self.dataset_img)
        self.num_img = len(self.datalist_img)
        self.num_vid = len(self.datalist_vid)
        self.shift_motion_model = True
        self.ScaleFactor = 1.0
        assert self.ScaleFactor == 1
        self.mean_value_ = [117, 117, 117]
        if cfg.DATASET.CATDOGHORSETK.FLAG_TEMPLATESEARCH_SIZEFROMCONFIG:
            self.resizedw_temp = cfg.TRAIN.EXEMPLAR_SIZE
            self.resizedh_temp = cfg.TRAIN.EXEMPLAR_SIZE
            self.resizedw_search = cfg.TRAIN.SEARCH_SIZE
            self.resizedh_search = cfg.TRAIN.SEARCH_SIZE
        else:
            self.resizedw_temp = cfg.TRAIN.SEARCH_SIZE
            self.resizedh_temp = cfg.TRAIN.SEARCH_SIZE
            self.resizedw_search = cfg.TRAIN.SEARCH_SIZE
            self.resizedh_search = cfg.TRAIN.SEARCH_SIZE

        self.lambda_shift_ = 5
        self.lambda_scale_ = 15
        self.lambda_min_scale_ = -0.4
        self.lambda_max_scale_ = 0.4
        self.kContextFactorShiftBox = cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTORSHIFTBOX

    def get_detection_image_bboxes(self, idx):
        det_imgpath = self.dataset_img[idx]
        det_bboxes = self.det_bbox[det_imgpath] # [[xmin, ymin, xmax, ymax, cid], ...]
        det_img = cv2.imread(det_imgpath+"1")

        if not isinstance(det_img, np.ndarray):
            print(22222)
        return det_img, det_bboxes



    def getnumdata(self, datalist, num):
        if num>0:
            n = len(datalist)
            if n > num:
                random.shuffle(datalist)
                return datalist[:num]
            else:
                d = []
                m = 0
                while m < num:
                    random.shuffle(datalist)
                    d.extend(datalist)
                    m = len(d)
                return d[:num]
        else:
            return datalist



    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = 127
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def get_otherobj(self):
        dsettag, rootid, imgpath_list, box_list = random.choice(self.datalist_other)
        idframe = random.randint(0,len(imgpath_list)-1)
        image_path = os.path.join(self.root_list[rootid], imgpath_list[idframe])
        image = cv2.imread(image_path)
        box = box_list[idframe]
        box = self._get_bbox(image,box)
        xmin,ymin,xmax,ymax = map(int,box)
        return image,[xmin,ymin,xmax,ymax]




    def calc_coverage(self, box1, box2):
        x1_1 = box1[0]
        y1_1 = box1[1]
        x2_1 = box1[2]
        y2_1 = box1[3]
        x1_2 = box2[0]
        y1_2 = box2[1]
        x2_2 = box2[2]
        y2_2 = box2[3]
        area_inter = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max((0, min(y2_1, y2_2) - max(y1_1, y1_2)))
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        return float(area_inter) / float(area1 + 1e-8), float(area_inter) / float(area2 + 1e-8)

    def putimg_on_template(self,imgback,imgper):
        shift_scale = 0.05
        aspect_scale = 0.05
        hback,wback = imgback.shape[:2]
        x_center = w = wback//2
        y_center = h = hback//2
        x_center += (random.uniform(0, 1.0) - 0.5) * 2 * shift_scale * w
        y_center += (random.uniform(0, 1.0) - 0.5) * 2 * shift_scale * h
        w *= ((random.uniform(0, 1.0) - 0.5) * 2 * aspect_scale + 1.0)
        h *= ((random.uniform(0, 1.0) - 0.5) * 2 * aspect_scale + 1.0)
        w,h = map(int,[w,h])
        imgper = cv2.resize(imgper,(w,h))
        x1 = x_center - w//2
        y1 = y_center - h//2
        x2 = x1 + w
        y2 = y1 + h
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        imgback[y1:y2,x1:x2,:] = imgper

        return imgback,[x1,y1,x2,y2]

    def putimg_on_search(self,imgback,imgper):
        aspect_scale = 0.15
        hback,wback = imgback.shape[:2]
        scale = float(self.resizedw_search)/float(self.resizedw_temp)*cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
        wa = float(wback/scale)
        ha = float(hback/scale)
        wa *= ((random.uniform(0, 1.0) - 0.5) * 2 * aspect_scale + 1.0)
        ha *= ((random.uniform(0, 1.0) - 0.5) * 2 * aspect_scale + 1.0)
        wa, ha = map(int, [wa, ha])

        xa_1 = random.randint(0, wback - wa - 1)
        ya_1 = random.randint(0, hback - ha - 1)
        xa_2 = xa_1 + wa
        ya_2 = ya_1 + ha
        boxa = [xa_1, ya_1, xa_2, ya_2]
        xa_1, ya_1, xa_2, ya_2 = map(int,[xa_1, ya_1, xa_2, ya_2])

        imgper = cv2.resize(imgper, (wa, ha))
        imgback[ya_1:ya_2,xa_1:xa_2,:] = imgper
        return imgback,boxa
    def __len__(self):
        return self.num

    def __getitem__(self, index):
        idximg = index % self.num_img
        # idxvid = index % self.num_vid
        iddetimg = index % self.num_det_img
        imgs_prev_all = []
        imgs_curr_all = []
        boxes_gt_all = []
        label_posnegpair = []


        ###step1:get negative pairs of other object
        # for i in range(cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHEROBJECT):
        #     imageother, box_other = self.get_otherobj()
        #     I_currs, box_gts,_ = self.MakeTrainingExamples(1, imageother, box_other)
        #     imgs_curr_all.append(I_currs[0])
        #     boxes_gt_all.append(box_gts[0])
        #     label_posnegpair.append(0)
        if random.uniform(0,1.0)<1.0:
            ###step2: get positive pairs of imagedata
            dsettag, rootid, imgpath, bbox_prev = self.datalist_img[idximg]
            a = os.path.join(self.root_list[rootid], imgpath)
            img_prev = cv2.imread(os.path.join(self.root_list[rootid], imgpath))
            if dsettag=='got':
                bbox_prev = self._get_bbox(img_prev,bbox_prev)
            img_curr = img_prev.copy()
            bbox_curr = bbox_prev[:]
            I_prev, imgs_curr, bboxes_gt,_ = self.MakeExamples(cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_IMAGE,
                                                                img_prev, img_curr, bbox_prev, bbox_curr)
            imgs_curr_all.extend(imgs_curr)
            boxes_gt_all.extend(bboxes_gt)
            label_posnegpair.extend([1]*len(imgs_curr))

        else:
            ###step4:get positive pairs of videodata
            dsettag, rootid, imgpath_list, bbox_list = self.datalist_vid[idxvid]
            frame_id_prev = random.randint(0, len(imgpath_list) - 1)
            frame_id_curr = random.randint(0, len(imgpath_list) - 1)
            img_prev = cv2.imread(os.path.join(self.root_list[rootid], imgpath_list[frame_id_prev]))
            img_curr = cv2.imread(os.path.join(self.root_list[rootid], imgpath_list[frame_id_curr]))
            bbox_prev = bbox_list[frame_id_prev]
            bbox_curr = bbox_list[frame_id_curr]
            if dsettag == 'got':
                bbox_prev = self._get_bbox(img_prev, bbox_prev)
                bbox_curr = self._get_bbox(img_curr, bbox_curr)
            I_prev, imgs_curr, bboxes_gt,_ = self.MakeExamples(cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_FRAME,
                                                                img_prev, img_curr, bbox_prev, bbox_curr)
            imgs_curr_all.extend(imgs_curr)
            boxes_gt_all.extend(bboxes_gt)
            label_posnegpair.extend([1]*len(imgs_curr))

        I_prev = Distortion((I_prev))
        if cfg.DATASET.CATDOGHORSETK.FLIP_TEMPLATE and cfg.DATASET.CATDOGHORSETK.FLIP_TEMPLATE > np.random.random():
            I_prev = cv2.flip(I_prev, 1)
        if cfg.DATASET.CATDOGHORSETK.FLIPVER_TEMPLATE and cfg.DATASET.CATDOGHORSETK.FLIPVER_TEMPLATE > np.random.random():
            I_prev = cv2.flip(I_prev, 0)
        n = cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHEROBJECT+cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_IMAGE + cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_FRAME
        imgs_prev_all.extend([I_prev] * n)


        ###############################finish get data

        for i in range(len(imgs_curr_all)):
            imgs_curr_all[i] = Distortion(imgs_curr_all[i])
            if cfg.DATASET.CATDOGHORSETK.FLIP_SEARCH and cfg.DATASET.CATDOGHORSETK.FLIP_SEARCH > np.random.random():
                imgs_curr_all[i] = cv2.flip(imgs_curr_all[i], 1)
                x1, y1, x2, y2 = boxes_gt_all[i]
                x1_new = self.ScaleFactor - x2
                x2_new = self.ScaleFactor - x1
                boxes_gt_all[i] = [x1_new, y1, x2_new, y2]
            if cfg.DATASET.CATDOGHORSETK.FLIPVER_SEARCH and cfg.DATASET.CATDOGHORSETK.FLIPVER_SEARCH > np.random.random():
                imgs_curr_all[i] = cv2.flip(imgs_curr_all[i], 0)
                x1, y1, x2, y2 = boxes_gt_all[i]
                y1_new = self.ScaleFactor - y2
                y2_new = self.ScaleFactor - y1
                boxes_gt_all[i] = [x1, y1_new, x2, y2_new]
        assert len(label_posnegpair)==len(boxes_gt_all)
        cls_list = []
        delta_list = []
        searchboxes = []
        for i in range(len(label_posnegpair)):
            flagneg = label_posnegpair[i] == 0
            x1,y1,x2,y2 = boxes_gt_all[i]
            x1 *= self.resizedw_search
            x2 *= self.resizedw_search
            y1 *= self.resizedh_search
            y2 *= self.resizedh_search
            bbox = Corner(x1, y1, x2, y2)
            searchboxes.append([x1,y1,x2,y2])
            cls, delta = self.point_target(bbox, cfg.TRAIN.OUTPUT_SIZE, flagneg)
            cls_list.append(cls)
            delta_list.append(delta)

        d_img, d_bboxes = self.get_detection_image_bboxes(iddetimg)
        detection_img, detection_bboxes = crop_detection_img(d_img, d_bboxes)
        for d in detection_bboxes:
            # print(d[0])
            x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 0, 255), 5)
        print(cv2.imwrite("/home/lbycdy/work/siamban/siamban/%d.jpg"%iddetimg, detection_img))

        I_prev = I_prev.transpose((2,0,1))

        search = np.stack(imgs_curr_all).transpose((0, 3, 1, 2))
        cls = np.stack(cls_list)
        delta = np.stack(delta_list)
        searchboxes = np.array(searchboxes)
        detection = detection_img.transpose((2, 0, 1)).astype(np.float32)
        return {
            'template': I_prev,
            'search': search,
            'label_cls': cls,
            'label_loc': delta,
            'searchboxes':searchboxes,
            'label_posnegpair': label_posnegpair,
            'detection': detection,
            'detection_bboxes': detection_bboxes
        }
    def MakeExamples(self, num_generated_examples, image_prev, image_curr, bbox_prev, bbox_curr,seggt=None):
        imgs_curr = []
        bboxes_gt = []
        seggt_list = []
        I_prev = self.getprevimg(image_prev, bbox_prev)
        I_curr, box_gt,seg = self.MakeTrueExample(image_curr, bbox_prev, bbox_curr,seggt)
        imgs_curr.append(I_curr)
        bboxes_gt.append(box_gt)
        seggt_list.append(seg)
        I_currs, box_gts,segs = self.MakeTrainingExamples(num_generated_examples - 1, image_curr, bbox_curr,seggt)
        imgs_curr.extend(I_currs)
        bboxes_gt.extend(box_gts)
        seggt_list.extend(segs)
        return I_prev, imgs_curr, bboxes_gt, seggt_list

    def getprevimg(self, image_prev, bbox_prev):
        shift_scale = 0.05
        aspect_scale = 0.05
        x1, y1, x2, y2 = bbox_prev
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        x_center += (random.uniform(0, 1.0) - 0.5) * 2 * shift_scale * w
        y_center += (random.uniform(0, 1.0) - 0.5) * 2 * shift_scale * h
        w *= ((random.uniform(0, 1.0) - 0.5) * 2 * aspect_scale + 1.0)
        h *= ((random.uniform(0, 1.0) - 0.5) * 2 * aspect_scale + 1.0)
        if cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 0:
            w = (x2 - x1) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
            h = (y2 - y1) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
        elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 1:
            box_w = x2 - x1
            box_h = y2 - y1
            w = box_w + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
            h = box_h + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
        elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 2:
            w = (x2 - x1) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
            h = (y2 - y1) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
            w = max(w, h)
            h = w
        elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 3:
            box_w = x2 - x1
            box_h = y2 - y1
            w = box_w + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
            h = box_h + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
            w = max(w, h)
            h = w

        roi_x1 = x_center - w / 2.0
        roi_x2 = x_center + w / 2.0
        roi_y1 = y_center - h / 2.0
        roi_y2 = y_center + h / 2.0
        roi_box = [roi_x1, roi_y1, roi_x2, roi_y2]

        imgcrop = self.getcropedimage(image_prev, roi_box,self.resizedw_temp,self.resizedh_temp)
        return imgcrop

    def MakeTrainingExamples(self, num_generated_examples, image_curr, bbox_curr,seggt=None):
        imgs_curr = []
        bboxes_gt = []
        seg_crop = []
        for i in range(num_generated_examples):
            if cfg.DATASET.CATDOGHORSETK.SHIFT_MODE == 0:
                bbox_rand = self.shift_bbox(image_curr, bbox_curr)
                x1, y1, x2, y2 = bbox_rand
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                if cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 0:
                    w = (x2 - x1) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
                    h = (y2 - y1) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
                elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 1:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    w = box_w + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
                    h = box_h + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
                elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 2:
                    w = (x2 - x1) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
                    h = (y2 - y1) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
                    w = max(w, h)
                    h = w
                elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 3:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    w = box_w + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
                    h = box_h + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
                    w = max(w, h)
                    h = w
            elif cfg.DATASET.CATDOGHORSETK.SHIFT_MODE == 1:
                crop_box = self.shift_bbox_remo(bbox_curr)
                xmin, ymin, xmax, ymax = crop_box
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin
            scale = float(self.resizedw_search) / float(self.resizedw_temp)
            w *= scale
            h *= scale
            roi_x1 = x_center - w / 2.0
            roi_x2 = x_center + w / 2.0
            roi_y1 = y_center - h / 2.0
            roi_y2 = y_center + h / 2.0
            roi_box = [roi_x1, roi_y1, roi_x2, roi_y2]
            I_curr = self.getcropedimage(image_curr, roi_box,self.resizedw_search,self.resizedh_search)
            imgs_curr.append(I_curr)
            if seggt is not None:
                seg_curr = self.getcropedimage(seggt, roi_box,self.resizedw_search,self.resizedh_search)
                seg_crop.append(seg_curr)
            gt_x1, gt_y1, gt_x2, gt_y2 = bbox_curr
            gt_x1 -= roi_x1
            gt_x2 -= roi_x1
            gt_y1 -= roi_y1
            gt_y2 -= roi_y1
            gt_x1 = gt_x1 / (roi_x2 - roi_x1 + 1e-8)
            gt_x2 = gt_x2 / (roi_x2 - roi_x1 + 1e-8)
            gt_y1 = gt_y1 / (roi_y2 - roi_y1 + 1e-8)
            gt_y2 = gt_y2 / (roi_y2 - roi_y1 + 1e-8)
            gt_x1 *= self.ScaleFactor
            gt_y1 *= self.ScaleFactor
            gt_x2 *= self.ScaleFactor
            gt_y2 *= self.ScaleFactor
            bboxes_gt.append([gt_x1, gt_y1, gt_x2, gt_y2])
        if seggt is not None:
            return imgs_curr, bboxes_gt, seg_crop
        else:
            return imgs_curr, bboxes_gt, [None, ]

    def MakeTrueExample(self, image_curr, bbox_prev, bbox_curr,seggt=None):

        x1, y1, x2, y2 = bbox_prev
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        if cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 0:
            w = (x2 - x1) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
            h = (y2 - y1) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
        elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 1:
            box_w = x2 - x1
            box_h = y2 - y1
            w = box_w + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
            h = box_h + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
        elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 2:
            w = (x2 - x1) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
            h = (y2 - y1) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
            w = max(w, h)
            h = w
        elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 3:
            box_w = x2 - x1
            box_h = y2 - y1
            w = box_w + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
            h = box_h + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
            w = max(w, h)
            h = w
        scale = float(self.resizedw_search)/float(self.resizedw_temp)
        w *= scale
        h *= scale
        roi_x1 = x_center - w / 2.0
        roi_x2 = x_center + w / 2.0
        roi_y1 = y_center - h / 2.0
        roi_y2 = y_center + h / 2.0
        roi_box = [roi_x1, roi_y1, roi_x2, roi_y2]
        img_curr = self.getcropedimage(image_curr, roi_box,self.resizedw_search,self.resizedh_search)
        gt_x1, gt_y1, gt_x2, gt_y2 = bbox_curr
        gt_x1 -= roi_x1
        gt_x2 -= roi_x1
        gt_y1 -= roi_y1
        gt_y2 -= roi_y1
        gt_x1 = gt_x1 / (roi_x2 - roi_x1 + 1e-8)
        gt_x2 = gt_x2 / (roi_x2 - roi_x1 + 1e-8)
        gt_y1 = gt_y1 / (roi_y2 - roi_y1 + 1e-8)
        gt_y2 = gt_y2 / (roi_y2 - roi_y1 + 1e-8)
        gt_x1 *= self.ScaleFactor
        gt_y1 *= self.ScaleFactor
        gt_x2 *= self.ScaleFactor
        gt_y2 *= self.ScaleFactor
        if seggt is not None:
            segcrop = self.getcropedimage(seggt, roi_box,self.resizedw_search,self.resizedh_search)
            return img_curr, [gt_x1, gt_y1, gt_x2, gt_y2], segcrop

        else:
            return img_curr, [gt_x1, gt_y1, gt_x2, gt_y2], None

    def getcropedimage(self, image, bbox,targetw,targeth):
        imgdim = image.shape
        imgh, imgw = imgdim[:2]
        x1, y1, x2, y2 = map(int, bbox)
        wbox = x2 - x1
        hbox = y2 - y1
        if wbox > 0 and hbox > 0:
            if len(imgdim) == 3:
                imgback = np.zeros((hbox, wbox, 3)).astype(np.uint8)
                for ic in range(3):
                    imgback[:, :, ic] = self.mean_value_[ic]
            else:
                imgback = np.zeros((hbox, wbox)).astype(np.uint8)

            xmin = max(x1, 0)
            ymin = max(y1, 0)
            xmax = min(x2, imgw)
            ymax = min(y2, imgh)
            if len(imgdim) == 3:
                imgonorg = image[ymin:ymax, xmin:xmax, :]
            else:
                imgonorg = image[ymin:ymax, xmin:xmax]

            worg = xmax - xmin
            horg = ymax - ymin
            if worg > 0 and horg > 0:
                xminonbak = xmin - x1
                yminonbak = ymin - y1
                if len(imgdim) == 3:
                    imgback[yminonbak:(yminonbak + horg), xminonbak:(xminonbak + worg), :] = imgonorg
                else:
                    imgback[yminonbak:(yminonbak + horg), xminonbak:(xminonbak + worg)] = imgonorg
            imgcrop = cv2.resize(imgback, (targetw, targeth))
            return imgcrop

        else:
            if len(imgdim) == 3:
                return np.zeros((targeth, targetw, 3)).astype(np.uint8)
            else:
                return np.zeros((targeth, targetw)).astype(np.uint8)

    #模仿bbox随机运动后的位置
    def shift_bbox(self, image, box):  # box = [xmin,ymin,xmax,ymax]
        """
         self.lambda_shift_ = 5
        self.lambda_scale_ = 15
        self.lambda_min_scale_ = -0.4
        self.lambda_max_scale_ = 0.4
        """
        img_h, img_w = image.shape[:2]
        width = box[2] - box[0]
        height = box[3] - box[1]
        center_x = (box[2] + box[0]) / 2.0
        center_y = (box[3] + box[1]) / 2.0
        kMaxNumTries = 10
        num_tries_width = 0
        new_width = -1
        while ((new_width < 0 or new_width > img_w - 1) and num_tries_width < kMaxNumTries):
            if self.shift_motion_model:
                smp_d = self.sample_exp_two_sided(self.lambda_scale_)
                width_scale_factor = max(self.lambda_min_scale_, min(self.lambda_max_scale_, smp_d))
            else:
                rand_num = random.random()
                width_scale_factor = self.lambda_min_scale_ + rand_num(
                    self.lambda_max_scale_ - self.lambda_min_scale_)
            new_width = width * (1 + width_scale_factor)
            new_width = max(1.0, min(img_w - 1, new_width))
            num_tries_width += 1

        num_tries_height = 0
        new_height = -1
        while ((new_height < 0 or new_height > img_h - 1) and num_tries_height < kMaxNumTries):
            if self.shift_motion_model:
                smp_d = self.sample_exp_two_sided(self.lambda_scale_)
                height_scale_factor = max(self.lambda_min_scale_, min(self.lambda_max_scale_, smp_d))
            else:
                rand_num = random.random()
                height_scale_factor = self.lambda_min_scale_ + rand_num(
                    self.lambda_max_scale_ - self.lambda_min_scale_)
            new_height = height * (1 + height_scale_factor)
            new_height = max(1.0, min(img_h - 1, new_height))
            num_tries_height += 1

        first_time_x = True
        new_center_x = -1
        num_tries_x = 0
        while ((first_time_x or
                new_center_x < center_x - width * self.kContextFactorShiftBox / 2 or
                new_center_x > center_x + width * self.kContextFactorShiftBox / 2 or
                new_center_x - new_width / 2 < 0 or
                new_center_x + new_width / 2 > img_w) and
               num_tries_x < kMaxNumTries):
            if self.shift_motion_model:
                smp_d = self.sample_exp_two_sided(self.lambda_shift_)
                new_x_temp = center_x + width * smp_d
            else:
                rand_num = random.random()
                new_x_temp = center_x + rand_num * (2 * new_width) - new_width

            new_center_x = min(img_w - new_width / 2, max(new_width / 2, new_x_temp))
            first_time_x = False
            num_tries_x += 1

        first_time_y = True
        new_center_y = -1
        num_tries_y = 0
        while ((first_time_y or
                new_center_y < center_y - height * self.kContextFactorShiftBox / 2 or
                new_center_y > center_y + height * self.kContextFactorShiftBox / 2 or
                new_center_y - new_height / 2 < 0 or
                new_center_y + new_height / 2 > img_h) and
               num_tries_y < kMaxNumTries):
            if self.shift_motion_model:
                smp_d = self.sample_exp_two_sided(self.lambda_shift_)
                new_y_temp = center_y + height * smp_d
            else:
                rand_num = random.random()
                new_y_temp = center_y + rand_num * (2 * new_height) - new_height

            new_center_y = min(img_h - new_height / 2, max(new_height / 2, new_y_temp))
            first_time_y = False
            num_tries_y += 1
        if num_tries_width >= kMaxNumTries or num_tries_height >= kMaxNumTries or num_tries_x >= kMaxNumTries or num_tries_y >= kMaxNumTries:
            x1, y1, x2, y2 = box
        else:
            x1 = new_center_x - new_width / 2
            x2 = new_center_x + new_width / 2
            y1 = new_center_y - new_height / 2
            y2 = new_center_y + new_height / 2
        return [x1, y1, x2, y2]
    def coveA(self, box1, box2):
        x1_1 = box1[0]
        y1_1 = box1[1]
        x2_1 = box1[2]
        y2_1 = box1[3]
        x1_2 = box2[0]
        y1_2 = box2[1]
        x2_2 = box2[2]
        y2_2 = box2[3]
        area_inter = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max((0, min(y2_1, y2_2) - max(y1_1, y1_2)))
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        return float(area_inter) / float(area1 + 1e-8)

    def shift_bbox_remo(self, box):  # box = [xmin,ymin,xmax,ymax]
        """

        self.lambda_min_scale_ = -0.4
        self.lambda_max_scale_ = 0.4
        """
        xmin, ymin, xmax, ymax = box
        cp_w = (xmax - xmin) * 2
        cp_h = (ymax - ymin) * 2
        cx_box = (xmin + xmax) / 2
        cy_box = (ymax + ymin) / 2
        sx = self.lambda_min_scale_ + random.uniform(0, 1.0) * (self.lambda_max_scale_ - self.lambda_min_scale_)
        sy = self.lambda_min_scale_ + random.uniform(0, 1.0) * (self.lambda_max_scale_ - self.lambda_min_scale_)
        cp_w *= (sx + 1.0)
        cp_h *= (sy + 1.0)

        shiftleft = cx_box - cp_w
        shiftright = cx_box
        shifttop = cy_box - cp_h
        shiftbottom = cy_box
        shiftleft, shiftright, shifttop, shiftbottom = map(int, [shiftleft, shiftright, shifttop, shiftbottom])
        flagok = False
        for i in range(20):
            if shiftleft < shiftright:
                x1 = random.randint(shiftleft, shiftright)
            else:
                x1 = xmin
            if shifttop < shiftbottom:
                y1 = random.randint(shifttop, shiftbottom)
            else:
                y1 = ymin
            x2 = x1 + cp_w
            y2 = y1 + cp_h
            if self.coveA(box, [x1, y1, x2, y2]) > 0.5:
                flagok = True
                break
        if not flagok:
            x1 = cx_box - cp_w / 2
            y1 = cy_box - cp_h / 2
            y2 = x1 + cp_w
            x2 = y2 + cp_h
        return [x1, y1, x2, y2]
    def sample_exp_two_sided(self, lam):
        prob = random.random()
        if prob > 0.5:
            pos_or_neg = 1
        else:
            pos_or_neg = -1
        rand_uniform = random.random()
        return math.log(rand_uniform) / lam * pos_or_neg

dist_param_ = {
    "brightness_prob": 0.3,
    "brightness_delta": 10,
    "contrast_prob": 0.3,
    "contrast_lower": 0.8,
    "contrast_upper": 1.2,
    "hue_prob": 0.3,
    "hue_delta": 10,
    "saturation_prob": 0.3,
    "saturation_lower": 0.8,
    "saturation_upper": 1.2,
    "random_order_prob": 0
}

def Distortion(image):
    if dist_param_ is not None:
        image = RandomBrightness(image, dist_param_["brightness_prob"], dist_param_["brightness_delta"])
        image = RandomContrast(image, dist_param_["contrast_prob"], dist_param_["contrast_lower"],
                            dist_param_["contrast_upper"])
        image = RandomSaturation(image, dist_param_["saturation_prob"], dist_param_["saturation_lower"],
                              dist_param_["saturation_upper"])
        image = RandomHue(image, dist_param_["hue_prob"], dist_param_["hue_delta"])
    return image


def RandomBrightness(img, brightness_prob, brightness_delta):
    prob = random.uniform(0, 1.0)
    img = img.astype(np.float)
    if (prob < brightness_prob):
        assert brightness_delta >= 0
        delta = random.uniform(-brightness_delta, brightness_delta)
        img += delta
    img = np.maximum(img, 0)
    img = np.minimum(img, 255)
    img = img.astype(np.uint8)
    return img

def RandomContrast(img, contrast_prob, lower, upper):
    prob = random.uniform(0, 1.0)
    img = img.astype(np.float)
    if (prob < contrast_prob):
        assert upper >= lower
        assert lower >= 0
        delta = random.uniform(lower, upper)
        if abs(delta - 1.0) > 1e-3:
            img *= delta
    img = np.maximum(img, 0)
    img = np.minimum(img, 255)
    img = img.astype(np.uint8)
    return img

def RandomSaturation(img, saturation_prob, lower, upper):
    prob = random.uniform(0, 1.0)
    img = img.astype(np.float32)
    if (prob < saturation_prob):
        assert upper >= lower
        assert lower >= 0
        delta = random.uniform(lower, upper)
        if abs(delta - 1.0) > 1e-3:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv[:, :, 1] *= delta
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img = np.maximum(img, 0)
    img = np.minimum(img, 255)
    img = img.astype(np.uint8)
    return img

def RandomHue(img, hue_prob, hue_delta):
    prob = random.uniform(0, 1.0)
    img = img.astype(np.float32)
    if (prob < hue_prob):
        assert hue_delta >= 0
        delta = random.uniform(-hue_delta, hue_delta)
        if abs(delta) > 0:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv[:, :, 0] += delta
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img = np.maximum(img, 0)
    img = np.minimum(img, 255)
    img = img.astype(np.uint8)
    return img

if __name__ == '__main__':
    from siamban.core.config import cfg
    cfg.DATASET.NAMES = ('REMOCATDOGHORSE',)
    cfg.DATASET.REMOCATDOGHORSE.ROOT = '/home/lbycdy/datasets/DogCatHorse/'
    cfg.DATASET.REMOCATDOGHORSE.ANNO = '/home/lbycdy/datasets/DogCatHorse/train_until20221201(1).json'
    cfg.TRAIN.DETECTION_SIZE = 160

    # img = cv2.imread('/home/lmn/Datasets/coco/train2017/000000019624.jpg')
    # cv2.namedWindow("img", cv2.NORM_HAMMING)
    # cv2.imshow('img', img)
    data = BANDataset()
    for i in range(2):
        a = data.__getitem__(i)
