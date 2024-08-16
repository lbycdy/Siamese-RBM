# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os
import time

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
import torch
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

class PointTarget1:
    def __init__(self, ):
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)

    def __call__(self, target, size, neg=False):

        # -1 ignore 0 negative 1 positive
        cls = 0 * np.ones((size, size), dtype=np.int64)
        delta = np.zeros((4, size, size), dtype=np.float32)
        delta_weight = np.zeros((1, size, size), dtype=np.float32)

        tcx, tcy, tw, th = corner2center(target)
        points = self.points.points

        if neg:
            return cls, delta

        delta[0] = points[0] - target[0]
        delta[1] = points[1] - target[1]
        delta[2] = target[2] - points[0]
        delta[3] = target[3] - points[1]

        # ellipse label
        pos = np.where(np.square(tcx - points[0]) / np.square(tw / cfg.POINT.SCALE_POS) +
                       np.square(tcy - points[1]) / np.square(th / cfg.POINT.SCALE_POS) < 1)
        # pos_num = len(pos[0])
        # delta_weight[0,pos] = 1. / (pos_num + 1e-6)
        cls[pos] = 1

        return cls, delta

class PointTarget2:
    def __init__(self, ):
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)

    def __call__(self, target, size, neg=False):

        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((size, size), dtype=np.int64)
        delta = np.zeros((4, size, size), dtype=np.float32)

        tcx, tcy, tw, th = corner2center(target)
        points = self.points.points

        if neg:
            neg = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                           np.square(tcy - points[1]) / np.square(th / 4) < 1)
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


        cls[pos] = 1
        cls[neg] = 0

        return cls, delta

class PointTarget3:
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
        numpos = pos[0].shape[0]
        numnegall = neg[0].shape[0]
        numnegsel = min(numnegall,int(numpos*cfg.TRAIN.NEGPOS_RATIO))
        neg, neg_num = select(neg, numnegsel)

        cls[pos] = 1
        cls[neg] = 0

        return cls, delta

class PointTarget_CLS:
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

            return cls

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
        numpos = pos[0].shape[0]
        numnegall = neg[0].shape[0]
        numnegsel = min(numnegall,int(numpos*cfg.TRAIN.NEGPOS_RATIO))
        neg, neg_num = select(neg, numnegsel)

        cls[pos] = 1
        cls[neg] = 0

        return cls
class PointTarget_LOC:
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

            return delta

        delta[0] = points[0] - target[0]
        delta[1] = points[1] - target[1]
        delta[2] = target[2]\
                   - points[0]
        delta[3] = target[3] - points[1]

        # ellipse label
        pos = np.where(np.square(tcx - points[0]) / np.square(tw / cfg.POINT.SCALE_POS) +
                       np.square(tcy - points[1]) / np.square(th / cfg.POINT.SCALE_POS) < 1)
        neg = np.where(np.square(tcx - points[0]) / np.square(tw / cfg.POINT.SCALE_NEG) +
                       np.square(tcy - points[1]) / np.square(th / cfg.POINT.SCALE_NEG) > 1)

        # sampling
        numpos = pos[0].shape[0]
        numnegall = neg[0].shape[0]
        numnegsel = min(numnegall,int(numpos*cfg.TRAIN.NEGPOS_RATIO))
        neg, neg_num = select(neg, numnegsel)

        cls[pos] = 1
        cls[neg] = 0

        return delta

class BANDataset(Dataset):
    def __init__(self,world_size=None,batchsize=None):
        super(BANDataset, self).__init__()
        self.path_format = '{}.{}.{}.jpg'

        # create point target
        if cfg.DATASET.CATDOGHORSETK.POSNEG_MODE==1:
            self.point_target = PointTarget1()
        elif cfg.DATASET.CATDOGHORSETK.POSNEG_MODE==2:
            self.point_target = PointTarget2()
        elif cfg.DATASET.CATDOGHORSETK.POSNEG_MODE == 3:
            self.point_target = PointTarget3()
            if cfg.DATASET.CATDOGHORSETK.CROPBB:
                self.point_target_cls = PointTarget_CLS()
                self.point_target_loc = PointTarget_LOC()
        else:
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
        self.datalist_headnoface = []
        self.root_list = []
        self.catlist_img = []
        self.doglist_img = []
        self.horselist_img = []
        self.catlist_vid = []
        self.doglist_vid = []
        self.horselist_vid = []
        rootid = 0
        cntboxhead = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            anno = subdata_cfg.ANNO
            if name not in ["HeadNOFACECOCO", "HeadNOFACEAIC"]:
                num_use_dog = subdata_cfg.NUM_USE_DOG
                num_use_cat = subdata_cfg.NUM_USE_CAT
                num_use_horse = subdata_cfg.NUM_USE_HORSE
                num_use_other = subdata_cfg.NUM_USE_OTHER

            data = json.load(open(anno))
            cat_list = []
            dog_list = []
            horse_list = []
            other_list = []

            if name in ["HeadNOFACECOCO","HeadNOFACEAIC"]:
                for imgpath in data:
                    boxinfo = [rootid,imgpath,data[imgpath]]
                    self.datalist_headnoface.append(boxinfo)
                    cntboxhead += len(data[imgpath])
                info1 = "{} {} images; {} boxes.".format(name, len(self.datalist_headnoface), cntboxhead)
                print(info1)
                logger.info(info1)
            elif name in ["REMOWIKICATHIGDOGANNORESULT20230427","REMOWIKICATHIGDOGANNORESULT20230504","REMOWIKICATHIGDOGANNORESULT20230508"]:
                for imgpath in data:
                    for box in data[imgpath]:
                        xmin, ymin, xmax, ymax, cid = box
                        boxinfo = ['remo', rootid, imgpath, [xmin, ymin, xmax, ymax],cid]

                        if cid == 1:

                            cat_list.append(boxinfo)
                        elif cid == 2:
                            dog_list.append(boxinfo)
                        elif cid == 3:
                            horse_list.append(boxinfo)
            else:
                for video in data:
                    for track in data[video]:
                        imgpath_list = []
                        box_list = []
                        if name == 'LASOT':
                            c = os.path.dirname(video)
                        elif name == 'GOT10K':
                            c = data[video][track]['cls'][1].strip()
                        elif name == 'YOUTUBEBB':
                            c = data[video][track]['cls'].strip()
                        else:
                            c = data[video][track]['cls']
                        c = str(c)
                        frames = data[video][track]
                        # frames = list(map(int,filter(lambda x: x.isdigit(), frames.keys())))
                        frames = filter(lambda x: x.isdigit(), frames.keys())

                        for frame in frames:
                            box = data[video][track][frame]
                            imgpath = os.path.join(video, self.path_format.format(frame, track, 'x'))
                            imgpath1 = os.path.join('/home/lbycdy/datasets',name,'crop511',imgpath)
                            img = cv2.imread(imgpath1)
                            # if img == None:
                            #     print(imgpath)
                            #     print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                            imgpath_list.append(imgpath)
                            box_list.append(box)

                        if c in self.clstag2clsid_dict:
                            if name in ['COCO', 'DET']:

                                _ = ['got', rootid, imgpath_list[0], box_list[0],self.clstag2clsid_dict[c]]
                            else:
                                boxinfo = ['got', rootid, imgpath_list, box_list,self.clstag2clsid_dict[c]]
                                if self.clstag2clsid_dict[c] == 1:
                                    cat_list.append(boxinfo)
                                elif self.clstag2clsid_dict[c] == 2:
                                    dog_list.append(boxinfo)
                                elif self.clstag2clsid_dict[c] == 3:
                                    horse_list.append(boxinfo)
                        else:
                            boxinfo = ['got', rootid, imgpath_list, box_list]
                            other_list.append(boxinfo)

            cat_list = self.getnumdata(cat_list, num_use_cat)
            dog_list = self.getnumdata(dog_list, num_use_dog)
            horse_list = self.getnumdata(horse_list, num_use_horse)
            other_list = self.getnumdata(other_list, num_use_other)
            if name in ['REMOWIKICATHIGDOGANNORESULT20230427','REMOWIKICATHIGDOGANNORESULT20230504','REMOWIKICATHIGDOGANNORESULT20230508']:
                if len(cat_list) > 0:
                    self.datalist_img.extend(cat_list)
                    self.catlist_img.extend(cat_list)
                if len(dog_list) > 0:
                    self.datalist_img.extend(dog_list)
                    self.doglist_img.extend(dog_list)
                if len(horse_list) > 0:
                    self.datalist_img.extend(horse_list)
                    self.horselist_img.extend(horse_list)
            else:
                if len(cat_list) > 0:
                    self.datalist_vid.extend(cat_list)
                    self.catlist_vid.extend(cat_list)
                if len(dog_list) > 0:
                    self.datalist_vid.extend(dog_list)
                    self.doglist_vid.extend(dog_list)
                if len(horse_list) > 0:
                    self.datalist_vid.extend(horse_list)
                    self.horselist_vid.extend(horse_list)

            if len(other_list) > 0:
                self.datalist_other.extend(other_list)
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
        self.world_size = torch.cuda.device_count()
        self.id_list = [random.choice([1, 2, 3]) for _ in range(int(self.num / self.world_size))]
        # self.id_list = [random.choice([1, 3]) for _ in range(int(self.num / self.world_size))]


        if world_size is not None and batchsize is not None:
            if self.num % (batchsize * world_size) != 0:
                n = self.num // (batchsize * world_size)
                self.num = (n + 1) * (batchsize * world_size)
        print(self.num,":num")
        random.shuffle(self.datalist_img)
        random.shuffle(self.datalist_vid)
        random.shuffle(self.datalist_other)
        random.shuffle(self.catlist_img)
        random.shuffle(self.doglist_img)
        random.shuffle(self.horselist_img)
        random.shuffle(self.catlist_vid)
        random.shuffle(self.doglist_vid)
        random.shuffle(self.horselist_vid)
        self.num_img = len(self.datalist_img)
        self.num_vid = len(self.datalist_vid)
        self.num_cat_img = len(self.catlist_img)
        self.num_dog_img = len(self.doglist_img)
        self.num_horse_img = len(self.horselist_img)

        self.num_cat_vid = len(self.catlist_vid)
        self.num_dog_vid = len(self.doglist_vid)
        self.num_horse_vid = len(self.horselist_vid)
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

        self.lambda_shift_ = cfg.DATASET.CATDOGHORSETK.LAMBDA_SHIFT #
        self.lambda_scale_ = cfg.DATASET.CATDOGHORSETK.LAMBDA_SCALE #15.0
        self.lambda_min_scale_ = cfg.DATASET.CATDOGHORSETK.LAMBDA_MIN_SCALE#-0.5
        self.lambda_max_scale_ = cfg.DATASET.CATDOGHORSETK.LAMBDA_MAX_SCALE#0.5
        self.lambda_min_ratio_ = cfg.DATASET.CATDOGHORSETK.LAMBDA_MIN_RATIO#-0.3
        self.lambda_max_ratio_ = cfg.DATASET.CATDOGHORSETK.LAMBDA_MAX_RATIO#0.3
        self.cova_thresh_shift_ = cfg.DATASET.CATDOGHORSETK.COVEA #0.5
        self.kContextFactorShiftBox = cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTORSHIFTBOX


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


    def putheadonsearch(self,imagehead,boxhead,imagesearch,boxtarget):
        x1,y1,x2,y2 = boxtarget
        xmint,ymint,xmaxt,ymaxt = x1*self.resizedw_search,y1*self.resizedh_search,x2*self.resizedw_search,y2*self.resizedh_search
        xmint, ymint, xmaxt, ymaxt = map(int,[xmint,ymint,xmaxt,ymaxt])
        xmint = max(0,xmint)
        ymint = max(0,ymint)
        xmaxt = min(xmaxt,self.resizedw_search)
        ymaxt = min(ymaxt,self.resizedh_search)
        wt = xmaxt - xmint
        ht = ymaxt - ymint

        imgh, imgw = imagehead.shape[:2]
        xmin, ymin, xmax, ymax = boxhead
        bw = xmax - xmin
        bh = ymax - ymin
        xmin -= 0.1 * bw
        xmax += 0.1 * bw
        ymin -= 0.1 * bh
        ymax += 0.1 * bh
        xmin, ymin, xmax,ymax = map(int,[xmin, ymin, xmax,ymax])
        xmin = max(0,xmin)
        ymin = max(0,ymin)
        xmax = min(xmax,imgw)
        ymax = min(ymax,imgh)
        imagehead = imagehead[ymin:ymax,xmin:xmax,:]
        size_target = int(random.uniform(0.5,1.5)*self.resizedw_temp/2)
        imgh,imgw = imagehead.shape[:2]
        scale = min(float(size_target)/imgh,float(size_target)/imgw)
        w = max(1,int(imgw*scale))
        h = max(1,int(imgh*scale))
        imagehead = cv2.resize(imagehead,(w,h))
        if wt>0 and ht > 0:
            flagok = False
            for i in range(20):
                x1 = random.randint(0, self.resizedw_search - w)
                y1 = random.randint(0, self.resizedh_search - h)
                x2 = x1 + w
                y2 = y1 + h
                if self.calc_coverage([x1,y1,x2,y2],[xmint,ymint,xmaxt,ymaxt])[0]<0.5:
                    flagok = True
                    break
            if flagok:
                imgt = imagesearch[ymint:ymaxt,xmint:xmaxt].copy()
                imagesearch[y1:y2, x1:x2] = imagehead
                imagesearch[ymint:ymaxt, xmint:xmaxt] = imgt
        else:
            x1 = random.randint(0,self.resizedw_search-w)
            y1 = random.randint(0,self.resizedh_search-h)
            x2 = x1 + w
            y2 = y1 + h
            imagesearch[y1:y2,x1:x2] = imagehead
        return imagesearch



    def comparetime(self,t1,t2):
        return int(t1[1:])>=int(t2[1:])
    def __len__(self):
        return self.num

    def __getitem__(self, index):

        idximg = index%self.num_img
        # idxvid = index%self.num_vid
        idxcatimg = index%(self.num_cat_img-4)
        idxdogimg = index%(self.num_dog_img-4)
        idxhorseimg = index%(self.num_horse_img-4)
        idxcatvid = index%(self.num_cat_vid-4)
        idxdogvid = index%(self.num_dog_vid-4)
        idxhorsevid = index%(self.num_horse_vid-4)
        imgs_prev_all = []
        imgs_curr_all = []
        boxes_gt_all = []
        label_posnegpair = []
        imgs_prev_temp = []
        img_vid_flg = []
        indtmp = index // self.world_size

        indret = self.id_list[indtmp]


        # print('#############',indret)
        ###step1:get negative pairs of other object
        for i in range(cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHEROBJECT):
            if random.uniform(0,1.0)<cfg.DATASET.CATDOGHORSETK.PROB_NEGFROM_HEADNOFACE:
                rootid,imgpath,boxes = random.choice(self.datalist_headnoface)
                imageother = cv2.imread(os.path.join(self.root_list[rootid],imgpath))
                box_other = random.choice(boxes)

            else:
                imageother, box_other = self.get_otherobj()
            I_currs, box_gts,_ = self.MakeTrainingExamples(1, imageother, box_other)
            imgs_curr_all.append(I_currs[0])
            boxes_gt_all.append(box_gts[0])
            label_posnegpair.append(0)
            img_vid_flg.append(1)
        for i in range(cfg.DATASET.CATDOGHORSETK.NBATCH_SHAPREPREV):
            if random.uniform(0,1.0)<cfg.DATASET.CATDOGHORSETK.PROB_IMG:

                ###step2: get positive pairs of imagedata
                # if i == 0:
                #     dsettag, rootid, imgpath, bbox_prev,clstag2clsid_dict = self.datalist_img[idximg]
                # else:
                #     if clstag2clsid_dict == 1:
                #         dsettag, rootid, imgpath, bbox_prev, _ = self.catlist_img[idxcatimg+i]
                #     elif clstag2clsid_dict == 2:
                #         dsettag, rootid, imgpath, bbox_prev, _ = self.doglist_img[idxdogimg+i]
                #     elif clstag2clsid_dict == 3:
                #         dsettag, rootid, imgpath, bbox_prev, _ = self.horselist_img[idxhorseimg+i]
                if indret == 1:
                    dsettag, rootid, imgpath, bbox_prev, _ = self.catlist_img[idxcatimg+i]
                elif indret == 2:
                    dsettag, rootid, imgpath, bbox_prev, _ = self.doglist_img[idxdogimg+i]
                elif indret == 3:
                    dsettag, rootid, imgpath, bbox_prev, _ = self.horselist_img[idxhorseimg+i]
                img_prev = cv2.imread(os.path.join(self.root_list[rootid], imgpath))
                if dsettag=='got':
                    bbox_prev = self._get_bbox(img_prev,bbox_prev)
                img_curr = img_prev.copy()
                bbox_curr = bbox_prev[:]
                I_prev, imgs_curr, bboxes_gt, _ = self.MakeExamples(cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_IMAGE,img_prev, img_curr, bbox_prev, bbox_curr)
                imgs_prev_temp.append(I_prev)
                imgs_curr_all.extend(imgs_curr)
                boxes_gt_all.extend(bboxes_gt)
                label_posnegpair.extend([1]*len(imgs_curr))
                img_vid_flg.extend(([1]*len(imgs_curr)))
            else:
                ###step4:get positive pairs of videodata
                # if i == 0:
                #
                #     dsettag, rootid, imgpath_list, bbox_list, clstag2clsid_dict = self.datalist_vid[idxvid]
                # else:
                #
                #     if clstag2clsid_dict == 1:
                #         dsettag, rootid, imgpath_list, bbox_list, _ = self.catlist_vid[idxcatvid+i]
                #     elif clstag2clsid_dict == 2:
                #         dsettag, rootid, imgpath_list, bbox_list, _ = self.doglist_vid[idxdogvid+i]
                #     elif clstag2clsid_dict == 3:
                #         dsettag, rootid, imgpath_list, bbox_list, _ = self.horselist_vid[idxhorsevid+i]
                if indret == 1:
                    dsettag, rootid, imgpath_list, bbox_list, _ = self.catlist_vid[idxcatvid+i]
                elif indret == 2:
                    dsettag, rootid, imgpath_list, bbox_list, _ = self.doglist_vid[idxdogvid+i]
                elif indret == 3:
                    dsettag, rootid, imgpath_list, bbox_list, _ = self.horselist_vid[idxhorsevid+i]

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
                imgs_prev_temp.append(I_prev)
                imgs_curr_all.extend(imgs_curr)
                boxes_gt_all.extend(bboxes_gt)
                label_posnegpair.extend([1]*len(imgs_curr))
                img_vid_flg.extend(([0] * len(imgs_curr)))
        # print('################',len(imgs_prev_temp))
        I_prev_num = np.random.choice((0,1,2,3))

        I_prev = imgs_prev_temp[I_prev_num]
        if random.uniform(0,1.0)<cfg.DATASET.CATDOGHORSETK.PROB_CROPTEMPLATE:
            sizehalf = self.resizedw_temp/2
            sx = random.uniform(cfg.DATASET.CATDOGHORSETK.SCALE_CROPTEMPLATE,1.0)
            sy = random.uniform(cfg.DATASET.CATDOGHORSETK.SCALE_CROPTEMPLATE,1.0)
            w_target = int(sizehalf*sx)
            h_target = int(sizehalf*sy)
            # if cfg.TRAIN.CODE_TIME_TAGE=='T20221213':
            if w_target < sizehalf:
                xmin = 40 + random.randint(0, sizehalf - w_target - 1)
            else:
                xmin = 40
            if h_target < sizehalf:
                ymin = 40 + random.randint(0, sizehalf - h_target - 1)
            else:
                ymin = 40

            xmax = xmin + w_target
            ymax = ymin + h_target
            w4 = w_target // 2
            h4 = h_target // 2
            xmin -= w4
            xmax += w4
            ymin -= h4
            ymax += h4
            xmin = max(0,xmin)
            ymin = max(0,ymin)
            xmax = min(self.resizedw_temp,xmax)
            ymax = min(self.resizedw_temp,ymax)
            I_prev = I_prev[ymin:ymax,xmin:xmax,:]
            I_prev = cv2.resize(I_prev,(self.resizedw_temp,self.resizedw_temp))

        for i in range(cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHEROBJECT,len(imgs_curr_all)):
            if random.uniform(0,1.0)<cfg.DATASET.CATDOGHORSETK.PROB_PUTHEADONSEARCH:
                rootid, imgpath, boxes = random.choice(self.datalist_headnoface)
                imagehead = cv2.imread(os.path.join(self.root_list[rootid], imgpath))
                boxhead = random.choice(boxes)
                imgs_curr_all[i] = self.putheadonsearch(imagehead, boxhead, imgs_curr_all[i], boxes_gt_all[i])

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
            flagneg = label_posnegpair[i]==0
            x1,y1,x2,y2 = boxes_gt_all[i]
            x1 *= self.resizedw_search
            x2 *= self.resizedw_search
            y1 *= self.resizedh_search
            y2 *= self.resizedh_search
            if cfg.DATASET.CATDOGHORSETK.CROPBB:
                x1_crop = max(0, x1)
                y1_crop = max(0, y1)
                x2_crop = min(x2, self.resizedw_search)
                y2_crop = min(y2, self.resizedh_search)
                bbox_crop = Corner(x1_crop, y1_crop, x2_crop, y2_crop)

                x1_crop = int(x1_crop)
                x2_crop = int(x2_crop)
                y1_crop = int(y1_crop)
                y2_crop = int(y2_crop)
                if random.uniform(0, 1) <= cfg.DATASET.CATDOGHORSETK.ERASE:
                    imgs_curr_all[i][y1_crop:y2_crop ,x1_crop:x2_crop,: ] = RandomErasing(imgs_curr_all[i][y1_crop:y2_crop,x1_crop:x2_crop,:],cfg.DATASET.CATDOGHORSETK.ERASE_MIN,cfg.DATASET.CATDOGHORSETK.ERASE_MAX, cfg.DATASET.CATDOGHORSETK.ERASE_RATIO)

                bbox = Corner(x1, y1, x2, y2)
                searchboxes.append([x1,y1,x2,y2])
                # print(x1,y1,x2,y2)
                cls, _ = self.point_target(bbox_crop, cfg.TRAIN.OUTPUT_SIZE, flagneg)
                # print("######",cls[0].shape())
                _, delta = self.point_target(bbox, cfg.TRAIN.OUTPUT_SIZE, flagneg)
            else:
                bbox = Corner(x1, y1, x2, y2)
                searchboxes.append([x1, y1, x2, y2])
                cls, delta = self.point_target(bbox, cfg.TRAIN.OUTPUT_SIZE, flagneg)
            cls_list.append(cls)
            delta_list.append(delta)
        I_prev = I_prev.transpose((2,0,1))
        search = np.stack(imgs_curr_all).transpose((0, 3, 1, 2))
        cls = np.stack(cls_list)
        delta = np.stack(delta_list)
        searchboxes = np.array(searchboxes)
        # print('######################',img_vid_flg.dtype())
        # print('######################1', label_posnegpair.dtype())
        # print('######################2', searchboxes.dtype())
        return {
            'template': I_prev,
            'search': search,
            'label_cls': cls,
            'label_loc': delta,
            'searchboxes': searchboxes,
            'label_posnegpair': label_posnegpair,
            'img_vid_flag': img_vid_flg,
            'indret': indret
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
        w, h, x_center, y_center = 0, 0, 0, 0
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
            elif cfg.DATASET.CATDOGHORSETK.SHIFT_MODE == 2:
                crop_box = self.shift_bbox_remo2(bbox_curr)
                xmin, ymin, xmax, ymax = crop_box
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin
            elif cfg.DATASET.CATDOGHORSETK.SHIFT_MODE == 6:
                crop_box = self.shift_bbox_remo6(bbox_curr)
                xmin, ymin, xmax, ymax = crop_box
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin
            elif cfg.DATASET.CATDOGHORSETK.SHIFT_MODE == 7:
                crop_box = self.shift_bbox_remo7(bbox_curr)
                xmin, ymin, xmax, ymax = crop_box
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin
            elif cfg.DATASET.CATDOGHORSETK.SHIFT_MODE == 8:
                crop_box = self.shift_bbox_remo8(bbox_curr)
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

    def sample_exp_two_sided(self, lam):
        prob = random.random()
        if prob > 0.5:
            pos_or_neg = 1
        else:
            pos_or_neg = -1
        rand_uniform = random.random()
        return math.log(rand_uniform) / lam * pos_or_neg

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
        bw = xmax - xmin
        bh = ymax - ymin
        cp_w = bw * 2 * float(self.resizedw_search)/float(self.resizedw_temp)
        cp_h = bh * 2 * float(self.resizedw_search)/float(self.resizedw_temp)

        cx_box = (xmin + xmax) / 2
        cy_box = (ymax + ymin) / 2

        s_rand = self.lambda_min_scale_ + random.uniform(0, 1.0) * (self.lambda_max_scale_ - self.lambda_min_scale_)
        r_rand = self.lambda_min_ratio_ + random.uniform(0, 1.0) * (self.lambda_max_ratio_ - self.lambda_min_ratio_)
        cp_w *= (1.0 + r_rand)
        cp_h /= (1.0 + r_rand)
        cp_w *= (1.0 + s_rand)
        cp_h *= (1.0 + s_rand)
        # print(cp_w/(bw*2),cp_h/(bh*2))

        shiftleft = (cx_box - bw/4) - cp_w
        shiftright = cx_box + bw/4
        shifttop = cy_box - bh/4 - cp_h
        shiftbottom = cy_box + bh/4

        # if self.comparetime(cfg.TRAIN.CODE_TIME_TAGE, 'T20230109'):
        #     shiftleft = cx_box - cp_w
        #     shiftright = cx_box + (xmax - xmin) * (0.5 - cfg.DATASET.CATDOGHORSETK.COVEA)
        #     shifttop = cy_box - cp_h
        #     shiftbottom = cy_box + (ymax - ymin) * (0.5 - cfg.DATASET.CATDOGHORSETK.COVEA)
        # else:
        #     shiftleft = cx_box - cp_w
        #     shiftright = cx_box
        #     shifttop = cy_box - cp_h
        #     shiftbottom = cy_box

        shiftleft, shiftright, shifttop, shiftbottom = map(int, [shiftleft, shiftright, shifttop, shiftbottom])
        flagok = False
        for i in range(50):
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
            if self.coveA(box, [x1, y1, x2, y2]) > self.cova_thresh_shift_:
                flagok = True
                break
        if not flagok:
            x1 = cx_box - cp_w / 2
            y1 = cy_box - cp_h / 2
            x2 = x1 + cp_w
            y2 = y1 + cp_h
        return [x1, y1, x2, y2]

    def shift_bbox_remo2(self,box):  # box = [xmin,ymin,xmax,ymax]
        """

        self.lambda_min_scale_ = -0.4
        self.lambda_max_scale_ = 0.4
        """
        xmin, ymin, xmax, ymax = box
        bw = xmax - xmin
        bh = ymax - ymin
        cp_w = bw * 2
        cp_h = bh * 2
        cx_box = (xmin + xmax) / 2
        cy_box = (ymax + ymin) / 2

        s_rand = self.lambda_min_scale_ + random.uniform(0, 1.0) * (self.lambda_max_scale_ - self.lambda_min_scale_)
        r_rand = self.lambda_min_ratio_ + random.uniform(0, 1.0) * (self.lambda_max_ratio_ - self.lambda_min_ratio_)
        cp_w *= (1.0 + r_rand)
        cp_h /= (1.0 + r_rand)
        cp_w *= (1.0 + s_rand)
        cp_h *= (1.0 + s_rand)
        cp_w = max(cp_w,bw)
        cp_h = max(cp_h,bh)


        shiftleft = xmax - cp_w
        shiftright = xmin
        shifttop = ymax - cp_h
        shiftbottom = ymin
        shiftleft, shiftright, shifttop, shiftbottom = map(int, [shiftleft, shiftright, shifttop, shiftbottom])
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
        return [x1, y1, x2, y2]




    def shift_bbox_remo6(self,box):  # box = [xmin,ymin,xmax,ymax]
        """

        self.lambda_min_scale_ = -0.4
        self.lambda_max_scale_ = 0.4
        """
        xmin, ymin, xmax, ymax = box
        cp_w = (xmax - xmin) * 2
        cp_h = (ymax - ymin) * 2
        cx_box = (xmin + xmax) / 2
        cy_box = (ymax + ymin) / 2
        r_rand = self.lambda_min_ratio_ + random.uniform(0, 1.0) * (self.lambda_max_ratio_ - self.lambda_min_ratio_)
        smp_d = self.sample_exp_two_sided(self.lambda_scale_)
        s_rand = max(self.lambda_min_scale_, min(self.lambda_max_scale_, smp_d))

        cp_w *= (1.0 + r_rand)
        cp_h /= (1.0 + r_rand)
        cp_w *= (1.0 + s_rand)
        cp_h *= (1.0 + s_rand)

        shiftleft = cx_box - cp_w
        shiftright = cx_box
        shifttop = cy_box - cp_h
        shiftbottom = cy_box
        shiftleft, shiftright, shifttop, shiftbottom = map(int, [shiftleft, shiftright, shifttop, shiftbottom])
        flagok = False
        for i in range(50):
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
            if self.coveA(box, [x1, y1, x2, y2]) > self.cova_thresh_shift_:
                flagok = True
                break
        if not flagok:
            x1 = cx_box - cp_w / 2
            y1 = cy_box - cp_h / 2
            x2 = x1 + cp_w
            y2 = y1 + cp_h
        return [x1, y1, x2, y2]

    def shift_bbox_remo7(self, box):  # box = [xmin,ymin,xmax,ymax]
        """

        self.lambda_min_scale_ = -0.4
        self.lambda_max_scale_ = 0.4
        """
        xmin, ymin, xmax, ymax = box
        bw = xmax - xmin
        bh = ymax - ymin
        if cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 0:
            cp_w = bw * 2 * float(self.resizedw_search) / float(self.resizedw_temp)  #
            cp_h = bh * 2 * float(self.resizedw_search) / float(self.resizedw_temp)
        elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 1:
            cp_w = bw + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (bw + bh)
            cp_h = bh + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (bw + bh)

        cx_box = (xmin + xmax) / 2
        cy_box = (ymax + ymin) / 2

        s_rand = self.lambda_min_scale_ + random.uniform(0, 1.0) * (self.lambda_max_scale_ - self.lambda_min_scale_)
        r_rand = self.lambda_min_ratio_ + random.uniform(0, 1.0) * (self.lambda_max_ratio_ - self.lambda_min_ratio_)
        cp_w *= (1.0 + r_rand)
        cp_h /= (1.0 + r_rand)
        cp_w *= (1.0 + s_rand)
        cp_h *= (1.0 + s_rand)
        # print(cp_w/(bw*2),cp_h/(bh*2))

        shiftleft = xmin - cp_w
        shiftright = xmax
        shifttop = ymin - cp_h
        shiftbottom = ymax

        shiftleft, shiftright, shifttop, shiftbottom = map(int, [shiftleft, shiftright, shifttop, shiftbottom])
        flagok = False
        for i in range(50):
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
            if self.coveA(box, [x1, y1, x2, y2]) > self.cova_thresh_shift_:
                flagok = True
                break
        if not flagok:
            x1 = cx_box - cp_w / 2
            y1 = cy_box - cp_h / 2
            x2 = x1 + cp_w
            y2 = y1 + cp_h
        return [x1, y1, x2, y2]


    def shift_bbox_remo8(self, box):  # box = [xmin,ymin,xmax,ymax]
        """

        self.lambda_min_scale_ = -0.4
        self.lambda_max_scale_ = 0.4
        """
        xmin, ymin, xmax, ymax = box
        bw = xmax - xmin
        bh = ymax - ymin
        # cp_w = bw * 2 * float(self.resizedw_search)/float(self.resizedw_temp)
        # cp_h = bh * 2 * float(self.resizedw_search)/float(self.resizedw_temp)

        cp_w = bw + 0.5 * (bw + bh)
        cp_h = bh + 0.5 * (bw + bh)

        cx_box = (xmin + xmax) / 2
        cy_box = (ymax + ymin) / 2

        s_rand = self.lambda_min_scale_ + random.uniform(0, 1.0) * (self.lambda_max_scale_ - self.lambda_min_scale_)
        r_rand = self.lambda_min_ratio_ + random.uniform(0, 1.0) * (self.lambda_max_ratio_ - self.lambda_min_ratio_)
        cp_w *= (1.0 + r_rand)
        cp_h /= (1.0 + r_rand)
        cp_w *= (1.0 + s_rand)
        cp_h *= (1.0 + s_rand)
        # print(cp_w/(bw*2),cp_h/(bh*2))

        shiftleft = xmin - cp_w
        shiftright = xmax
        shifttop = ymin - cp_h
        shiftbottom = ymax

        shiftleft, shiftright, shifttop, shiftbottom = map(int, [shiftleft, shiftright, shifttop, shiftbottom])
        flagok = False
        for i in range(50):
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
            if self.coveA(box, [x1, y1, x2, y2]) > self.cova_thresh_shift_:
                flagok = True
                break
        if not flagok:
            x1 = cx_box - cp_w / 2
            y1 = cy_box - cp_h / 2
            x2 = x1 + cp_w
            y2 = y1 + cp_h
        return [x1, y1, x2, y2]
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

def RandomErasing(img_bbox,sl,sh,r1):
    """ Randomly selects a rectangle region in an image and erases its pixels.
            'Random Erasing Data Augmentation' by Zhong et al.
            See https://arxiv.org/pdf/1708.04896.pdf
        Args:
             probability: The probability that the Random Erasing operation will be performed.
             sl: Minimum proportion of erased area against input image.
             sh: Maximum proportion of erased area against input image.
             r1: Minimum aspect ratio of erased area.
             mean: Erasing value.
        """

    area = img_bbox.shape[0] * img_bbox.shape[1]
    target_area = random.uniform(sl, sh) * area
    aspect_ratio = random.uniform(r1, 1 / r1)

    h = int(round(math.sqrt(target_area * aspect_ratio)))
    w = int(round(math.sqrt(target_area / aspect_ratio)))
    mean = np.random.randint(0, 255, (h,w,3))
    if w < img_bbox.shape[1] and h < img_bbox.shape[0]:
        y1 = random.randint(0, img_bbox.shape[0] - h)
        x1 = random.randint(0, img_bbox.shape[1] - w)
        if img_bbox.shape[2] == 3:
            img_bbox[ y1:y1 + h, x1:x1 + w, 0] = mean[:, :, 0]
            img_bbox[ y1:y1 + h, x1:x1 + w, 1] = mean[:, :, 1]
            img_bbox[ y1:y1 + h, x1:x1 + w, 2] = mean[:, :, 2]
        else:
            img_bbox[y1:y1 + h ,x1:x1 + w, 0] = mean[:, :, 0]
    return img_bbox

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
    cfgfile = "/home/lbycdy/work/siamban/experiments/siamban_r50_l234/test.yaml"
    cfg.merge_from_file(cfgfile)
    cfg.DATASET.CATDOGHORSETK.PROB_CROPTEMPLATE = 1.0
    data = BANDataset()
    idlist = list(range(10))
    random.shuffle(idlist)
    cnt = 0
    for i in idlist:
        a = data.__getitem__(i)
        # for k in a:
        #     print(k, a[k].shape)
        search = a["search"]
        template = a['template']
        boxes_gt_all = a['searchboxes']
        label_posnegpair = a['label_posnegpair']
        nimg = search.shape[0]
        t = template.transpose((1, 2, 0))
        cv2.namedWindow("imgt", cv2.NORM_MINMAX)
        cv2.imshow("imgt", t)
        for j in range(nimg):
            imgs = search[j].transpose((1, 2, 0))
            loc = boxes_gt_all[j]
            x1, y1, x2, y2 = loc
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if label_posnegpair[j] == 1:
                cv2.rectangle(imgs, (x1, y1), (x2, y2), (0, 0, 255))
            # else:
                # cv2.rectangle(imgs, (x1, y1), (x2, y2), (0, 255, 0))
            cv2.namedWindow("imgs", cv2.NORM_MINMAX)
            cv2.imshow("imgs", imgs)

            key = cv2.waitKey()
            if key == 27:
                exit()
