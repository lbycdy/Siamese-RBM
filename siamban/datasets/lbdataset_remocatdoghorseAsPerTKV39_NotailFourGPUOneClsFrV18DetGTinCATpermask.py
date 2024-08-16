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
from siamban.models.rank_loss import IoU
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


        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        points = self.points.points

        if neg:
            cls = -1 * np.ones((size, size), dtype=np.int64)
            delta = np.zeros((4, size, size), dtype=np.float32)
            for i in range(len(target)):
                xmin, ymin, xmax, ymax = target[i]
                tcx = (xmin+xmax)/2.0
                tcy = (ymin+ymax)/2.0
                tw = xmax - xmin
                th = ymax - ymin

                if i==0:
                    neg = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                           np.square(tcy - points[1]) / np.square(th / 4) < 1)
                else:
                    flag = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                           np.square(tcy - points[1]) / np.square(th / 4) < 1)
                    neg = np.bitwise_or(neg,flag)
            neg, neg_num = select(neg, cfg.TRAIN.NEG_NUM)
            cls[neg] = 0

            return cls, delta

        cls = -1 * np.ones((size, size), dtype=np.int64)
        delta_list = []
        pos = np.zeros((size,size),dtype=np.bool_)
        neg = np.ones((size,size),dtype=np.bool_)
        idxgtarea = -1*np.ones((2,size,size))
        for i in range(len(target)):
            delta = np.zeros((4, size, size), dtype=np.float32)
            xmin, ymin, xmax, ymax = target[i]
            tcx = (xmin + xmax) / 2.0
            tcy = (ymin + ymax) / 2.0
            tw = xmax - xmin
            th = ymax - ymin
            a = tw*th
            delta[0] = points[0] - target[i][0]
            delta[1] = points[1] - target[i][1]
            delta[2] = target[i][2] - points[0]
            delta[3] = target[i][3] - points[1]
            delta_list.append(delta)
            flagpos = np.square(tcx - points[0]) / np.square(tw / cfg.POINT.SCALE_POS) + np.square(tcy - points[1]) / np.square(th / cfg.POINT.SCALE_POS) < 1
            flagneg = np.square(tcx - points[0]) / np.square(tw / cfg.POINT.SCALE_NEG) +np.square(tcy - points[1]) / np.square(th / cfg.POINT.SCALE_NEG) > 1
            neg = np.bitwise_and(flagneg,neg)
            flagrepeatpos = np.bitwise_and(pos,flagpos)
            flaguniquepos = np.bitwise_xor(flagpos,flagrepeatpos)
            pos[flaguniquepos]=True
            idxgtarea[0,flaguniquepos]=i
            idxgtarea[1,flaguniquepos]=a
            areasrepeat = idxgtarea[1,flagrepeatpos]
            flagreplace = areasrepeat>a
            idxgtarea[0,flagrepeatpos][flagreplace]=i
            idxgtarea[1,flagrepeatpos][flagreplace]=a
        delta = np.zeros((4, size, size), dtype=np.float32)
        for i in range(len(target)):
            flag = idxgtarea[0]==i
            delta[:, flag] = delta_list[i][:, flag]
        pos = np.where(pos > 0)
        neg = np.where(neg > 0)
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
        self.datalist_otherper = []
        self.dataset_mask = []
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
                num_use_otherper = -1
            data = json.load(open(anno))
            cat_list = []
            dog_list = []
            horse_list = []
            other_list = []
            otherper_list = []
            permask_list = []
            # cat_vid_list = []
            # dog_vid_list = []
            # horse_vid_list = []

            if name in ["HeadNOFACECOCO","HeadNOFACEAIC"]:
                for imgpath in data:
                    boxinfo = [rootid,imgpath,data[imgpath]]
                    self.datalist_headnoface.append(boxinfo)
                    cntboxhead += len(data[imgpath])
                info1 = "{} {} images; {} boxes.".format(name, len(self.datalist_headnoface), cntboxhead)
                print(info1)
                logger.info(info1)
            elif name in ["REMANNONOTAILDATA230525230526IMAGE","REMANNONOTAILDATA230525230613IMAGE","REMANNONOTAILDATA0616IMAGE"]:
                cnthasother = 0
                cntcat = 0
                for imgpath in data:
                    for idxbox,box in enumerate(data[imgpath]):
                        xmin, ymin, xmax, ymax, cid = box
                        boxother = []
                        for idxj,boxb in enumerate(data[imgpath]):
                            cidj = boxb[-1]
                            if idxj != idxbox and cidj == cid:
                                boxother.append(boxb)
                        if len(boxother)>0 and cid == 1:
                            cnthasother += 1
                        if cid==1:
                            cntcat+=1
                        boxinfo = ['remoimg', rootid, imgpath, [xmin, ymin, xmax, ymax], cid,boxother]
                        if cid == 1:
                            cat_list.append(boxinfo)
                        elif cid == 2:
                            dog_list.append(boxinfo)
                        elif cid == 3:
                            horse_list.append(boxinfo)
                print(cntcat,cnthasother,":cnthasother")
            elif name in ["REMANNONOTAILDATA230525230526VIDEO", ]:
                for vid in data:
                    imgpath_list = []
                    box_list = []
                    for imgpath,box,cid in data[vid]:
                        imgpath_list.append(imgpath)
                        box_list.append(box)
                    if cid == 2:
                        boxinfo = ['removid', rootid, imgpath_list, box_list, 2]
                        dog_list.append(boxinfo)
                    elif cid == 1:
                        boxinfo = ['removid', rootid, imgpath_list, box_list, 1]
                        cat_list.append(boxinfo)
            elif name in ['LASOTNOTAIL1', 'GOT10KNOTAIL1', 'VIDNOTAIL1']:
                for vidtrackid in data:
                    vidid = vidtrackid.split("#")[0]

                    annoinfo = data[vidtrackid]
                    imgpath_list = []
                    box_list = []
                    catname = annoinfo[0]
                    imgboxinfo = annoinfo[1]
                    # print(catname)
                    for imgname, box in imgboxinfo:
                        imgpath = os.path.join(vidid, imgname)
                        imgpath_list.append(imgpath)
                        box_list.append(box)
                    if catname == 'horse':
                        boxinfo = ['removid', rootid, imgpath_list, box_list, 3]
                        horse_list.append(boxinfo)
                    elif catname == 'dog':
                        boxinfo = ['removid', rootid, imgpath_list, box_list, 2]
                        dog_list.append(boxinfo)
                    elif catname == 'cat':
                        boxinfo = ['removid', rootid, imgpath_list, box_list, 1]
                        cat_list.append(boxinfo)

            elif name in ["AICMASK", ]:
                for imgpath in data:
                    for name,box in data[imgpath]:
                        xmin, ymin, xmax, ymax = box
                        boxinfo = ['AIC', rootid, imgpath,name, [xmin, ymin, xmax, ymax]]
                        otherper_list.append(boxinfo)
                    # per_name = os.path.splitext(os.path.basename(imgpath))[0]

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
                        if c not in self.clstag2clsid_dict:
                            frames = data[video][track]
                            # frames = list(map(int,filter(lambda x: x.isdigit(), frames.keys())))
                            frames = filter(lambda x: x.isdigit(), frames.keys())
                            for frame in frames:
                                box = data[video][track][frame]
                                imgpath = os.path.join(video, self.path_format.format(frame, track, 'x'))
                                imgpath_list.append(imgpath)
                                box_list.append(box)
                            boxinfo = ['got', rootid, imgpath_list, box_list]
                            other_list.append(boxinfo)

            cat_list = self.getnumdata(cat_list, num_use_cat)
            dog_list = self.getnumdata(dog_list, num_use_dog)
            horse_list = self.getnumdata(horse_list, num_use_horse)
            other_list = self.getnumdata(other_list, num_use_other)
            otherper_list = self.getnumdata(otherper_list,num_use_otherper)
            permask_list = self.getnumdata(otherper_list,num_use_otherper)


            if name in ["REMANNONOTAILDATA230525230526IMAGE","REMANNONOTAILDATA230525230613IMAGE","REMANNONOTAILDATA0616IMAGE"]:
                if len(cat_list) > 0:
                    self.catlist_img.extend(cat_list)
                if len(dog_list) > 0:
                    self.doglist_img.extend(dog_list)
                if len(horse_list) > 0:
                    self.horselist_img.extend(horse_list)
            elif name in ['REMANNONOTAILDATA230525230526VIDEO','LASOTNOTAIL1', 'GOT10KNOTAIL1', 'VIDNOTAIL1']:
                if len(cat_list) > 0:
                    self.catlist_vid.extend(cat_list)
                if len(dog_list) > 0:
                    self.doglist_vid.extend(dog_list)
                if len(horse_list) > 0:
                    self.horselist_vid.extend(horse_list)

            if len(other_list) > 0:
                self.datalist_other.extend(other_list)
            if len(otherper_list) > 0:
                self.datalist_otherper.extend(otherper_list)
            if len(permask_list) > 0:
                self.dataset_mask.extend(permask_list)
            info1 = "{} {} catimg; {} dogimg; {} horseimg.".format(name,len(self.catlist_img),len(self.doglist_img),len(self.horselist_img))
            info2 = "{} {} catvid; {} dogvid; {} horsevid.".format(name,len(self.catlist_vid),len(self.doglist_vid),len(self.horselist_vid))
            info3 = "{} {} datalist_other;{} datalist_otherper;{} datalist_mask".format(name,len(self.datalist_other),len(self.datalist_otherper),len(self.dataset_mask))
            logger.info(info1)
            logger.info(info2)
            logger.info(info3)

            print(info1)
            print(info2)
            print(info3)

            root = subdata_cfg.ROOT
            self.root_list.append(root)
            rootid += 1

        self.videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = self.videos_per_epoch if self.videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.world_size = torch.cuda.device_count()
        self.id_list = [random.choice([1, 2, 3]) for _ in range(int(self.num / self.world_size))]
        # self.id_list = [random.choice([1, 3]) for _ in range(int(self.num / self.world_size))]
        self.num_catimg = len(self.catlist_img)
        self.num_dogimg = len(self.doglist_img)
        self.num_horseimg = len(self.horselist_img)
        self.num_catvid = len(self.catlist_vid)
        self.num_dogvid = len(self.doglist_vid)
        self.num_horsevid = len(self.horselist_vid)

        if world_size is not None and batchsize is not None:
            if self.num % (batchsize * world_size) != 0:
                n = self.num // (batchsize * world_size)
                self.num = (n + 1) * (batchsize * world_size)
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
        self.shuffledata()

    def shuffledata(self):
        self.datalist_pos = []
        self.datalist_neg = []
        self.datalist_per = []
        self.datalist_permask = []
        self.datalist_permask1 = []
        random.shuffle(self.catlist_img)
        random.shuffle(self.catlist_vid)
        random.shuffle(self.doglist_img)
        random.shuffle(self.doglist_vid)
        random.shuffle(self.horselist_img)
        random.shuffle(self.horselist_vid)
        random.shuffle(self.datalist_other)
        random.shuffle(self.datalist_otherper)
        cntcat_img = 0
        cntcat_vid = 0
        cntdog_img = 0
        cntdog_vid = 0
        cnthorse_img = 0
        cnthorse_vid = 0
        samplelist = cfg.DATASET.CATDOGHORSETK.CLASS_SAMPLELIST
        numiter = int(self.videos_per_epoch//4)
        for i in range(numiter):
            idcls = samplelist[i%len(samplelist)]
            for igpu in range(4):
                di = []
                for _ in range(cfg.DATASET.CATDOGHORSETK.FETCH_ITERS):
                    if random.uniform(0,1.0)<cfg.DATASET.CATDOGHORSETK.PROB_IMG:
                        if idcls==1:
                            di.append(self.catlist_img[cntcat_img%self.num_catimg])
                            cntcat_img += 1
                        elif idcls==2:
                            di.append(self.doglist_img[cntdog_img%self.num_dogimg])
                            cntdog_img += 1
                        else:
                            di.append(self.horselist_img[cnthorse_img%self.num_horseimg])
                            cnthorse_img += 1
                    else:
                        if idcls == 1:
                            di.append(self.catlist_vid[cntcat_vid%self.num_catvid])
                            cntcat_vid += 1
                        elif idcls == 2:
                            di.append(self.doglist_vid[cntdog_vid%self.num_dogvid])
                            cntdog_vid += 1
                        else:
                            di.append(self.horselist_vid[cnthorse_vid%self.num_horsevid])
                            cnthorse_vid += 1
                self.datalist_pos.append(di)
        cntother = 0
        cntotherper = 0
        cntpermask = 0
        cntpermask1 = 0
        for i in range(numiter):
            for igpu in range(4):
                di = []
                di_per = []
                di_permask = []
                di_permask1 = []
                for _ in range(cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHEROBJECT):
                    di.append(self.datalist_other[cntother%len(self.datalist_other)])
                    cntother+=1
                self.datalist_neg.append(di)
                for _ in range(cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHERPER):
                    di_per.append(self.datalist_otherper[cntotherper%len(self.datalist_otherper)])
                    cntotherper+=1
                self.datalist_per.append(di_per)
                for _ in range(cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_IMAGE - 1):
                    di_permask.append(self.dataset_mask[cntpermask%len(self.dataset_mask)])
                    cntpermask+=1
                self.datalist_permask.append(di_permask)
                for _ in range(cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_IMAGE):
                    di_permask1.append(self.dataset_mask[cntpermask1%len(self.dataset_mask)])
                    cntpermask1+=1
                self.datalist_permask1.append(di_permask1)


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
    def get_otherobj(self,datainfo):
        # dsettag, rootid, imgpath_list, box_list = random.choice(self.datalist_other)
        dsettag, rootid, imgpath_list, box_list = datainfo
        idframe = random.randint(0,len(imgpath_list)-1)
        image_path = os.path.join(self.root_list[rootid], imgpath_list[idframe])

        image = cv2.imread(image_path)

        box = box_list[idframe]
        # print(os.path.join(self.root_list[rootid], imgpath_list[idframe]))
        box = self._get_bbox(image,box)
        xmin,ymin,xmax,ymax = map(int,box)
        return image,[xmin,ymin,xmax,ymax]
    def get_otherobj1(self,datainfo):
        # dsettag, rootid, imgpath_list, box_list = random.choice(self.datalist_other)
        dsettag, rootid, imgpath_list,imgmask_list, box_list = datainfo
        image_path = os.path.join(self.root_list[rootid], imgpath_list)
        img = cv2.imread(os.path.join('/home/lbycdy/datasets/DogCatHorse/AIC_Data_20230630_orisize',imgmask_list))
        image = cv2.imread(image_path)
        # print(image_path)
        # print(os.path.join('/home/lbycdy/datasets/DogCatHorse/AIC_Data_20230630_orisize',imgmask_list))
        box = box_list
        # img1 = np.zeros_like(image)
        # img1[:, :, 0] = img[:, :, 0] * image[:, :, 0]
        # img1[:, :, 1] = img[:, :, 1] * image[:, :, 1]
        # img1[:, :, 2] = img[:, :, 2] * image[:, :, 2]
        # box = self._get_bbox(image,box)
        anno = np.zeros_like(box)
        anno[0] = box[0]*image.shape[1]
        anno[1] = box[1]*image.shape[0]
        anno[2] = box[2]*image.shape[1]
        anno[3] = box[3]*image.shape[0]
        xmin,ymin,xmax,ymax = map(int,anno)

        return img[ymin:ymax,xmin:xmax,:],image,[xmin,ymin,xmax,ymax]

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
        xmin, ymin, xmax,ymax = map(int, [xmin, ymin, xmax,ymax])
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax,imgw)
        ymax = min(ymax,imgh)
        imagehead = imagehead[ymin:ymax, xmin:xmax, :]
        size_target = int(random.uniform(0.5,1.5)*self.resizedw_temp/2)
        imgh,imgw = imagehead.shape[:2]
        scale = min(float(size_target)/imgh,float(size_target)/imgw)
        w = max(1,int(imgw*scale))
        h = max(1,int(imgh*scale))
        imagehead = cv2.resize(imagehead,(w,h))
        if wt > 0 and ht > 0:
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
        imgs_prev_all = []
        imgs_curr_all = []
        boxes_gt_all = []
        label_posnegpair = []
        imgs_prev_temp = []
        imgs_per = []
        boxes_per = []
        imgs_per1 = []
        boxes_per1 = []
        imgs_per_temp = []
        imgs_per_temp1 = []
        permask_temp = []
        permask_temp1 = []
        boxes_per_temp = []
        per_mask = []
        # print(len(self.datalist_pos),len(self.datalist_neg)
        # print('#############',indret)
        ###step1:get negative pairs of other object
        if cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHEROBJECT>0:
            for di in self.datalist_neg[index%self.videos_per_epoch]:
                imageother, box_other = self.get_otherobj(di)
                I_currs, box_gts, _ = self.MakeTrainingExamples(1, imageother, box_other)
                imgs_curr_all.append(I_currs[0])

                boxes_gt_all.append(box_gts[0])
                label_posnegpair.append(0)
        if cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHERPER>0:
            for di in self.datalist_per[index % self.videos_per_epoch]:

                permask,imageother, box_other = self.get_otherobj1(di)
                I_currs, box_gts, _ = self.MakeTrainingExamples(1, imageother, box_other)

                imgs_curr_all.append(I_currs[0])
                boxes_gt_all.append(box_gts[0])
                label_posnegpair.append(0)
        if cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHERPER>0:
            for di in self.datalist_permask[index % self.videos_per_epoch]:
                permask, imageother, box_other = self.get_otherobj1(di)
                imgs_per_temp.append(imageother)
                permask_temp.append(permask)
                boxes_per.append(box_other)
        if cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHERPER>0:
            for di in self.datalist_permask1[index % self.videos_per_epoch]:
                permask, imageother, box_other = self.get_otherobj1(di)
                imgs_per_temp1.append(imageother)
                permask_temp1.append(permask)
                boxes_per1.append(box_other)

        for i in range(len(imgs_per_temp)):
            imgper = imgs_per_temp[i][boxes_per[i][1]:boxes_per[i][3], boxes_per[i][0]:boxes_per[i][2],:]
            imgs_per.append(imgper)
        for i in range(len(imgs_per_temp1)):
            imgper1 = imgs_per_temp1[i][boxes_per1[i][1]:boxes_per1[i][3], boxes_per1[i][0]:boxes_per1[i][2],:]
            imgs_per1.append(imgper1)


        cidlist = []
        idxtemplate = -1
        for idx, dinfo in enumerate(self.datalist_pos[index%self.videos_per_epoch]):
            dsettag = dinfo[0]
            # print(dinfo)
            if dsettag=="remoimg":
                rootid, imgpath, bbox_prev, cid,boxes_other = dinfo[1:]
                img_prev = cv2.imread(os.path.join(self.root_list[rootid], imgpath))
                # print(os.path.join(self.root_list[rootid], imgpath))
                img_curr = img_prev.copy()
                bbox_curr = bbox_prev[:]
                p = random.uniform(0, 1.0)
                flagcroptemplate = (cid == 1 and p<cfg.DATASET.CATDOGHORSETK.PROB_CROPTEMPLATE_CAT) or (cid !=1 and p<cfg.DATASET.CATDOGHORSETK.PROB_CROPTEMPLATE_OTHER)
                if flagcroptemplate:
                    I_prev, imgs_curr, bboxes_gt, _ = self.MakeExamplesCropTemplate(
                        cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_IMAGE,
                        img_prev, img_curr, bbox_prev, bbox_curr,imgs_per1,boxes_per1,permask_temp1,boxesother=boxes_other)
                    idxtemplate = idx
                else:

                    I_prev, imgs_curr, bboxes_gt, _ = self.MakeExamples(cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_IMAGE,img_prev, img_curr, bbox_prev, bbox_curr,imgs_per,boxes_per,permask_temp,boxesother=boxes_other)
                imgs_prev_temp.append(I_prev)
                imgs_curr_all.extend(imgs_curr)

                boxes_gt_all.extend(bboxes_gt)
                label_posnegpair.extend([1] * len(imgs_curr))
            else:
                rootid, imgpath_list, bbox_list, cid = dinfo[1:]
                frame_id_prev = random.randint(0, len(imgpath_list) - 1)
                frame_id_curr = random.randint(0, len(imgpath_list) - 1)
                img_prev = cv2.imread(os.path.join(self.root_list[rootid], imgpath_list[frame_id_prev]))
                img_curr = cv2.imread(os.path.join(self.root_list[rootid], imgpath_list[frame_id_curr]))
                bbox_prev = bbox_list[frame_id_prev]
                bbox_curr = bbox_list[frame_id_curr]
                # print(os.path.join(self.root_list[rootid], imgpath_list[frame_id_curr]))

                I_prev, imgs_curr, bboxes_gt, _ = self.MakeExamples(cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_FRAME,
                                                                    img_prev, img_curr, bbox_prev, bbox_curr,imgs_per,boxes_per,permask_temp)
                imgs_prev_temp.append(I_prev)
                imgs_curr_all.extend(imgs_curr)
                boxes_gt_all.extend(bboxes_gt)
                label_posnegpair.extend([1] * len(imgs_curr))
            cidlist.append(cid)

        # print('################',len(imgs_prev_temp))
        if idxtemplate<0:
            I_prev_num = np.random.choice((0, 1, 2, 3))
            I_prev = imgs_prev_temp[I_prev_num]
        else:
            I_prev = imgs_prev_temp[idxtemplate]

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
        n = cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHEROBJECT+cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHERPER+cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_IMAGE + cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_FRAME
        imgs_prev_all.extend([I_prev] * n)
        # print('##############',n)

        ###############################finish get data

        for i in range(len(imgs_curr_all)):
            imgs_curr_all[i] = Distortion(imgs_curr_all[i])
            if cfg.DATASET.CATDOGHORSETK.FLIP_SEARCH and cfg.DATASET.CATDOGHORSETK.FLIP_SEARCH > np.random.random():
                imgs_curr_all[i] = cv2.flip(imgs_curr_all[i], 1)
                boxes = []
                for box in boxes_gt_all[i]:
                    x1, y1, x2, y2 = box
                    x1_new = self.ScaleFactor - x2
                    x2_new = self.ScaleFactor - x1
                    boxes.append([x1_new, y1, x2_new, y2])
                boxes_gt_all[i] = boxes
            if cfg.DATASET.CATDOGHORSETK.FLIPVER_SEARCH and cfg.DATASET.CATDOGHORSETK.FLIPVER_SEARCH > np.random.random():
                imgs_curr_all[i] = cv2.flip(imgs_curr_all[i], 0)
                boxes = []
                for box in boxes_gt_all[i]:
                    x1, y1, x2, y2 = box
                    y1_new = self.ScaleFactor - y2
                    y2_new = self.ScaleFactor - y1
                    boxes.append([x1, y1_new, x2, y2_new])
                boxes_gt_all[i] = boxes
        assert len(label_posnegpair) == len(boxes_gt_all)
        cls_list = []
        delta_list = []
        searchboxes = []
        for i in range(len(label_posnegpair)):
            flagneg = label_posnegpair[i] == 0
            boxes = []
            for box in boxes_gt_all[i]:
                x1,y1,x2,y2 = box
                x1 *= self.resizedw_search
                x2 *= self.resizedw_search
                y1 *= self.resizedh_search
                y2 *= self.resizedh_search
                boxes.append([x1, y1, x2, y2])
            cls, delta = self.point_target(boxes, cfg.TRAIN.OUTPUT_SIZE, flagneg)
            searchboxes.append(boxes[0])
            cls_list.append(cls)
            delta_list.append(delta)
        I_prev = I_prev.transpose((2, 0, 1))
        search = np.stack(imgs_curr_all).transpose((0, 3, 1, 2))
        # print('$############',len(search))
        cls = np.stack(cls_list)
        delta = np.stack(delta_list)
        searchboxes = np.array(searchboxes)
        return {
            'template': I_prev,
            'search': search,
            'label_cls': cls,
            'label_loc': delta,
            'searchboxes': searchboxes,
            'images_per': imgs_per,
            'boxes_per': boxes_per,
            'label_posnegpair': label_posnegpair,
            'cidlist': np.array(cidlist),
        }

    def MakeExamplesCropTemplate(self, num_generated_examples, image_prev, image_curr, bbox_prev, bbox_curr,imgs_per,boxes_per,permask_temp, seggt=None,boxesother=[]):
        imgs_curr = []
        bboxes_gt = []
        seggt_list = []
        I_prev,bbox_fake = self.getcropedprevimg(image_prev, bbox_prev)
        I_currs, box_gts, segs = self.MakeTrainingExamplesCropedTemplate(num_generated_examples, bbox_fake,image_curr, bbox_curr,imgs_per,boxes_per,permask_temp, seggt,boxesother)
        imgs_curr.extend(I_currs)
        bboxes_gt.extend(box_gts)
        seggt_list.extend(segs)
        return I_prev, imgs_curr, bboxes_gt, seggt_list

    def MakeExamples(self, num_generated_examples, image_prev, image_curr, bbox_prev, bbox_curr,imgs_per,boxes_per,permask_temp,seggt=None,boxesother=[]):
        imgs_curr = []
        bboxes_gt = []
        seggt_list = []
        I_prev = self.getprevimg(image_prev, bbox_prev)
        I_curr, box_gt,seg = self.MakeTrueExample(image_curr, bbox_prev, bbox_curr,seggt,boxesother)
        imgs_curr.append(I_curr)
        bboxes_gt.append(box_gt)
        seggt_list.append(seg)
        I_currs, box_gts,segs = self.MakeTrainingExamples1(num_generated_examples - 1, image_curr, bbox_curr,imgs_per,boxes_per,permask_temp,seggt,boxesother)
        imgs_curr.extend(I_currs)
        bboxes_gt.extend(box_gts)
        seggt_list.extend(segs)
        return I_prev, imgs_curr, bboxes_gt, seggt_list
    def getcropedprevimg(self, image_prev, bbox_prev):
        x1, y1, x2, y2 = bbox_prev
        wbox = x2 - x1
        hbox = y2 - y1
        sx = random.uniform(cfg.DATASET.CATDOGHORSETK.SCALE_CROPTEMPLATE, 0.95)
        sy = random.uniform(cfg.DATASET.CATDOGHORSETK.SCALE_CROPTEMPLATE, 0.95)
        w_target = int(wbox * sx)
        h_target = int(hbox * sy)
        x1fake = x1 + random.uniform(0,wbox-w_target)
        y1fake = y1 + random.uniform(0,hbox-h_target)
        x2fake = x1fake + w_target
        y2fake = y1fake + h_target
        if cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 0:
            w = (x2fake - x1fake) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
            h = (y2fake - y1fake) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
        elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 1:
            box_w = x2fake - x1fake
            box_h = y2fake - y1fake
            w = box_w + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
            h = box_h + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
        elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 2:
            w = (x2fake - x1fake) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
            h = (y2fake - y1fake) * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
            w = max(w, h)
            h = w
        elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 3:
            box_w = x2fake - x1fake
            box_h = y2fake - y1fake
            w = box_w + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
            h = box_h + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
            w = max(w, h)
            h = w
        x_center = (x1fake + x2fake)/2.0
        y_center = (y1fake + y2fake)/2.0
        roi_x1 = x_center - w / 2.0
        roi_x2 = x_center + w / 2.0
        roi_y1 = y_center - h / 2.0
        roi_y2 = y_center + h / 2.0
        roi_box = [roi_x1, roi_y1, roi_x2, roi_y2]

        imgcrop = self.getcropedimage(image_prev, roi_box,self.resizedw_temp,self.resizedh_temp)
        return imgcrop,[x1fake,y1fake,x2fake,y2fake]
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

    def MakeTrainingExamples1(self, num_generated_examples, image_curr, bbox_curr,imgs_per,boxes_per,permask_temp,seggt=None,boxesother=[]):
        imgs_curr = []
        bboxes_gt = []
        seg_crop = []
        w, h, x_center, y_center = 0, 0, 0, 0
        for i in range(num_generated_examples):
            if cfg.DATASET.CATDOGHORSETK.SHIFT_MODE == 1:
                crop_box = self.shift_bbox_remo(bbox_curr)
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

            scale = float(self.resizedw_search) / float(self.resizedw_temp)
            w *= scale
            h *= scale
            roi_x1 = x_center - w / 2.0
            roi_x2 = x_center + w / 2.0
            roi_y1 = y_center - h / 2.0
            roi_y2 = y_center + h / 2.0
            roi_box = [roi_x1, roi_y1, roi_x2, roi_y2]

            I_curr = self.getcropedimage(image_curr, roi_box,self.resizedw_search,self.resizedh_search)
            # imgs_curr.append(I_curr)
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
            boxes_ii = []
            boxes_ii.append([gt_x1, gt_y1, gt_x2, gt_y2])
            for box in boxesother:
                cov = self.coveA(box[:4], roi_box)
                if cov > self.cova_thresh_shift_:
                    xminother, yminother, xmaxother, ymaxother = box[:4]
                    xminother -= roi_x1
                    xmaxother -= roi_x1
                    yminother -= roi_y1
                    ymaxother -= roi_y1
                    xminother = xminother / (roi_x2 - roi_x1 + 1e-8)
                    xmaxother = xmaxother / (roi_x2 - roi_x1 + 1e-8)
                    yminother = yminother / (roi_y2 - roi_y1 + 1e-8)
                    ymaxother = ymaxother / (roi_y2 - roi_y1 + 1e-8)
                    xminother *= self.ScaleFactor
                    yminother *= self.ScaleFactor
                    xmaxother *= self.ScaleFactor
                    ymaxother *= self.ScaleFactor
                    boxes_ii.append([xminother, yminother, xmaxother, ymaxother])
            # print(boxes_per)
            if np.random.random() < cfg.DATASET.CATDOGHORSETK.PERMATTING and imgs_per[i].shape[0] != 0 and imgs_per[i].shape[1] != 0:
                flag_mat = False
                for k in range(100):
                    imageper = imgs_per[i]
                    maskper = permask_temp[i]
                    boxes_matting = [int(gt_x1 * 160), int(gt_y1 * 160),
                                     int(gt_x2 * 160), int(gt_y2 * 160)]
                    h_per = imageper.shape[0]
                    w_per = imageper.shape[1]

                    r = h_per / w_per
                    c = boxes_matting[2] - boxes_matting[0]
                    d = c * r
                    a = random.random()
                    b = random.random()
                    boxes_per_temp = boxes_per
                    boxes_per_temp[i][1] = a * 150
                    boxes_per_temp[i][0] = b * 150
                    boxes_per_temp[i][3] = int(boxes_per_temp[i][1] + d)
                    boxes_per_temp[i][2] = int(boxes_per_temp[i][0] + c)
                    boxes_per_temp[i][1] = int(boxes_per_temp[i][1])
                    boxes_per_temp[i][0] = int(boxes_per_temp[i][0])
                    boxes_per_temp[i][3] = min(boxes_per_temp[i][3], 160)
                    boxes_per_temp[i][2] = min(boxes_per_temp[i][2], 160)
                    w_per = boxes_per_temp[i][2] - boxes_per_temp[i][0]
                    h_per = boxes_per_temp[i][3] - boxes_per_temp[i][1]
                    imageper = cv2.resize(imageper, (w_per, h_per))
                    maskper = cv2.resize(maskper, (w_per, h_per))
                    boxes_matting[0] = max(boxes_matting[0], 0)
                    boxes_matting[1] = max(boxes_matting[1], 0)
                    boxes_matting[2] = min(boxes_matting[2], 160)
                    boxes_matting[3] = min(boxes_matting[3], 160)
                    if self.IoU(boxes_matting,boxes_per_temp[i % len(imgs_per)]) > 0.2 and self.IoU(boxes_matting,boxes_per_temp[i % len(imgs_per)]) <= 0.4:
                        img_curr_matting = I_curr[boxes_matting[1]:boxes_matting[3], boxes_matting[0]:boxes_matting[2],:]
                        img_curr_matting1 = img_curr_matting.copy()
                        nnn = np.ones_like(
                            I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3], boxes_per_temp[i][0]:boxes_per_temp[i][2],0])
                        I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3], boxes_per_temp[i][0]:boxes_per_temp[i][2],
                        0] = (nnn - maskper[:, :, 0]) * I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3],
                                                        boxes_per_temp[i][0]:boxes_per_temp[i][2], 0] + maskper[:, :,
                                                                                                        0] * imageper[:,
                                                                                                             :, 0]
                        I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3], boxes_per_temp[i][0]:boxes_per_temp[i][2],
                        1] = (nnn - maskper[:, :, 1]) * I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3],
                                                        boxes_per_temp[i][0]:boxes_per_temp[i][2], 1] + maskper[:, :,
                                                                                                        1] * imageper[:,
                                                                                                             :, 1]
                        I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3], boxes_per_temp[i][0]:boxes_per_temp[i][2],
                        2] = (nnn - maskper[:, :, 2]) * I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3],
                                                        boxes_per_temp[i][0]:boxes_per_temp[i][2], 2] + maskper[:, :,
                                                                                                        0] * imageper[:,
                                                                                                             :, 2]
                        I_curr[boxes_matting[1]:boxes_matting[3], boxes_matting[0]:boxes_matting[2],
                        :] = img_curr_matting1
                        imgs_curr.append(I_curr)
                        bboxes_gt.append(boxes_ii)
                        flag_mat == True
                        break
                if flag_mat == False:
                    imgs_curr.append(I_curr)
                    bboxes_gt.append(boxes_ii)
            else:
                imgs_curr.append(I_curr)
                bboxes_gt.append(boxes_ii)

        if seggt is not None:
            return imgs_curr, bboxes_gt, seg_crop
        else:
            return imgs_curr, bboxes_gt, [None, ]
    def MakeTrainingExamples(self, num_generated_examples, image_curr, bbox_curr,seggt=None,boxesother=[]):
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

            I_curr = self.getcropedimage(image_curr, roi_box, self.resizedw_search,self.resizedh_search)

            # imgs_curr.append(I_curr)
            if seggt is not None:
                seg_curr = self.getcropedimage(seggt, roi_box, self.resizedw_search,self.resizedh_search)
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
            boxes_ii = []
            boxes_ii.append([gt_x1, gt_y1, gt_x2, gt_y2])
            for box in boxesother:
                cov = self.coveA(box[:4], roi_box)
                if cov > self.cova_thresh_shift_:
                    xminother, yminother, xmaxother, ymaxother = box[:4]
                    xminother -= roi_x1
                    xmaxother -= roi_x1
                    yminother -= roi_y1
                    ymaxother -= roi_y1
                    xminother = xminother / (roi_x2 - roi_x1 + 1e-8)
                    xmaxother = xmaxother / (roi_x2 - roi_x1 + 1e-8)
                    yminother = yminother / (roi_y2 - roi_y1 + 1e-8)
                    ymaxother = ymaxother / (roi_y2 - roi_y1 + 1e-8)
                    xminother *= self.ScaleFactor
                    yminother *= self.ScaleFactor
                    xmaxother *= self.ScaleFactor
                    ymaxother *= self.ScaleFactor
                    boxes_ii.append([xminother, yminother, xmaxother, ymaxother])

            imgs_curr.append(I_curr)
            bboxes_gt.append(boxes_ii)

        if seggt is not None:
            return imgs_curr, bboxes_gt, seg_crop
        else:
            return imgs_curr, bboxes_gt, [None, ]
    def MakeTrainingExamplesCropedTemplate(self, num_generated_examples,bbox_fake,image_curr, bbox_curr,imgs_per,boxes_per,permask_temp,seggt=None,boxesother=[]):
        imgs_curr = []
        bboxes_gt = []
        seg_crop = []
        w, h, x_center, y_center = 0, 0, 0, 0
        for i in range(num_generated_examples):
            if i==0:
                xmin,ymin,xmax,ymax = bbox_fake
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                box_w = xmax - xmin
                box_h = ymax - ymin
                if cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 0:
                    w = box_w * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
                    h = box_h * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
                elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 1:
                    w = box_w + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
                    h = box_h + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
            else:
                xmin, ymin, xmax, ymax = bbox_fake
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                box_w = xmax - xmin
                box_h = ymax - ymin
                if cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 0:
                    w = box_w * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
                    h = box_h * cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR
                elif cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX == 1:
                    w = box_w + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
                    h = box_h + cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR * (box_w + box_h)
                xshift = (random.uniform(0.0,1.0)-0.5)*(w - box_w)
                yshift = (random.uniform(0.0,1.0)-0.5)*(h - box_h)
                x_center += xshift
                y_center += yshift
            scale = float(self.resizedw_search) / float(self.resizedw_temp)
            w *= scale
            h *= scale
            roi_x1 = x_center - w / 2.0
            roi_x2 = x_center + w / 2.0
            roi_y1 = y_center - h / 2.0
            roi_y2 = y_center + h / 2.0
            roi_box = [roi_x1, roi_y1, roi_x2, roi_y2]
            I_curr = self.getcropedimage(image_curr, roi_box,self.resizedw_search,self.resizedh_search)

            # imgs_curr.append(I_curr)
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
            boxes_ii = []
            boxes_ii.append([gt_x1, gt_y1, gt_x2, gt_y2])
            for box in boxesother:
                cov = self.coveA(box[:4], roi_box)
                if cov > self.cova_thresh_shift_:
                    xminother, yminother, xmaxother, ymaxother = box[:4]
                    xminother -= roi_x1
                    xmaxother -= roi_x1
                    yminother -= roi_y1
                    ymaxother -= roi_y1
                    xminother = xminother / (roi_x2 - roi_x1 + 1e-8)
                    xmaxother = xmaxother / (roi_x2 - roi_x1 + 1e-8)
                    yminother = yminother / (roi_y2 - roi_y1 + 1e-8)
                    ymaxother = ymaxother / (roi_y2 - roi_y1 + 1e-8)
                    xminother *= self.ScaleFactor
                    yminother *= self.ScaleFactor
                    xmaxother *= self.ScaleFactor
                    ymaxother *= self.ScaleFactor
                    boxes_ii.append([xminother, yminother, xmaxother, ymaxother])
            # bboxes_gt.append(boxes_ii)
            if np.random.random() < cfg.DATASET.CATDOGHORSETK.PERMATTING and imgs_per[i].shape[0] != 0 and imgs_per[i].shape[1] != 0:
                flag_mat = False
                for k in range(100):
                    imageper = imgs_per[i]
                    maskper = permask_temp[i]
                    boxes_matting = [int(gt_x1 * 160), int(gt_y1 * 160),
                                     int(gt_x2 * 160), int(gt_y2 * 160)]
                    h_per = imageper.shape[0]
                    w_per = imageper.shape[1]

                    r = h_per / w_per
                    c = boxes_matting[2] - boxes_matting[0]
                    d = c * r
                    a = random.random()
                    b = random.random()
                    boxes_per_temp = boxes_per
                    boxes_per_temp[i][1] = a * 150
                    boxes_per_temp[i][0] = b * 150
                    boxes_per_temp[i][3] = int(boxes_per_temp[i][1] + d)
                    boxes_per_temp[i][2] = int(boxes_per_temp[i][0] + c)
                    boxes_per_temp[i][1] = int(boxes_per_temp[i][1])
                    boxes_per_temp[i][0] = int(boxes_per_temp[i][0])
                    boxes_per_temp[i][3] = min(boxes_per_temp[i][3], 160)
                    boxes_per_temp[i][2] = min(boxes_per_temp[i][2], 160)
                    w_per = boxes_per_temp[i][2] - boxes_per_temp[i][0]
                    h_per = boxes_per_temp[i][3] - boxes_per_temp[i][1]
                    imageper = cv2.resize(imageper, (w_per, h_per))
                    maskper = cv2.resize(maskper,(w_per, h_per))
                    boxes_matting[0] = max(boxes_matting[0], 0)
                    boxes_matting[1] = max(boxes_matting[1], 0)
                    boxes_matting[2] = min(boxes_matting[2], 160)
                    boxes_matting[3] = min(boxes_matting[3], 160)
                    if self.IoU(boxes_matting, boxes_per_temp[i % len(imgs_per)]) > 0.2 and self.IoU(boxes_matting,boxes_per_temp[i % len(imgs_per)]) <= 0.4:
                        img_curr_matting = I_curr[boxes_matting[1]:boxes_matting[3], boxes_matting[0]:boxes_matting[2],
                                           :]
                        img_curr_matting1 = img_curr_matting.copy()
                        # cv2.imshow('@@@@@@@2', img_curr_matting)
                        nnn = np.ones_like(I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3], boxes_per_temp[i][0]:boxes_per_temp[i][2],0])
                        I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3], boxes_per_temp[i][0]:boxes_per_temp[i][2], 0] = (nnn-maskper[:,:,0]) * I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3], boxes_per_temp[i][0]:boxes_per_temp[i][2], 0]+maskper[:,:,0]*imageper[:,:,0]
                        I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3], boxes_per_temp[i][0]:boxes_per_temp[i][2],
                        1] = (nnn-maskper[:,:,1]) * I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3], boxes_per_temp[i][0]:boxes_per_temp[i][2], 1]+maskper[:,:,1]*imageper[:,:,1]
                        I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3], boxes_per_temp[i][0]:boxes_per_temp[i][2],
                        2] = (nnn-maskper[:,:,2]) * I_curr[boxes_per_temp[i][1]:boxes_per_temp[i][3], boxes_per_temp[i][0]:boxes_per_temp[i][2], 2]+maskper[:,:,0]*imageper[:,:,2]
                        # cv2.imshow('@@@@@@@3', I_curr)

                        I_curr[boxes_matting[1]:boxes_matting[3], boxes_matting[0]:boxes_matting[2],
                        :] = img_curr_matting1
                        # cv2.imshow('@@@@@@@4', I_curr)
                        imgs_curr.append(I_curr)
                        bboxes_gt.append(boxes_ii)
                        flag_mat = True
                        # cv2.imshow('44', I_curr)
                        # key = cv2.waitKey(0)
                        # if key == 27:
                        #     exit()
                        break
                if flag_mat == False:
                    imgs_curr.append(I_curr)
                    bboxes_gt.append(boxes_ii)
            else:
                imgs_curr.append(I_curr)
                bboxes_gt.append(boxes_ii)
        if seggt is not None:
            return imgs_curr, bboxes_gt, seg_crop
        else:
            return imgs_curr, bboxes_gt, [None, ]
    def MakeTrueExample(self, image_curr, bbox_prev, bbox_curr,seggt=None,boxesother=[]):
        boxes_gtall = []

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
        boxes_gtall.append([gt_x1, gt_y1, gt_x2, gt_y2])
        for box in boxesother:
            cov = self.coveA(box[:4], roi_box)
            if cov > self.cova_thresh_shift_:
                xminother,yminother,xmaxother,ymaxother = box[:4]
                xminother -= roi_x1
                xmaxother -= roi_x1
                yminother -= roi_y1
                ymaxother -= roi_y1
                xminother = xminother / (roi_x2 - roi_x1 + 1e-8)
                xmaxother = xmaxother / (roi_x2 - roi_x1 + 1e-8)
                yminother = yminother / (roi_y2 - roi_y1 + 1e-8)
                ymaxother = ymaxother / (roi_y2 - roi_y1 + 1e-8)
                xminother *= self.ScaleFactor
                yminother *= self.ScaleFactor
                xmaxother *= self.ScaleFactor
                ymaxother *= self.ScaleFactor
                boxes_gtall.append([xminother,yminother,xmaxother,ymaxother])


        if seggt is not None:
            segcrop = self.getcropedimage(seggt, roi_box,self.resizedw_search,self.resizedh_search)
            return img_curr, boxes_gtall, segcrop

        else:
            return img_curr, boxes_gtall, None

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

    def IoU(self,box1,box2):

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
        return float(area_inter) / float(area1 + area2 - area_inter + 1e-8)

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
    cfgfile = "/home/lbycdy/work/siamban/experiments/siamban_r50_l234/20230608.yaml"
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
        imgs_per = a['images_per']
        boxes_per = a['boxes_per']
        # for a in range(6):
        #     img = imgs_per[a]
        #     loc = boxes_per[a]
        #     x1, y1, x2, y2 = loc[0],loc[1],loc[2],loc[3]
        #     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
        #     cv2.namedWindow("imgs", cv2.NORM_MINMAX)
        #     cv2.imshow("imgs", img)
        #
        #     key = cv2.waitKey()
        #     if key == 27:
        #         exit()
        nimg = search.shape[0]
        t = template.transpose((1,2,0))
        cv2.namedWindow("imgt", cv2.NORM_MINMAX)
        cv2.imshow("imgt", t)
        for j in range(nimg):
            imgs = search[j].transpose((1, 2, 0))
            loc = boxes_gt_all[j]
            x1,y1,x2,y2 = loc
            x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
            if label_posnegpair[j]==1:
                cv2.rectangle(imgs,(x1,y1),(x2,y2),(0,0,255))
            else:
                cv2.rectangle(imgs,(x1,y1),(x2,y2),(0,255,0))
            cv2.namedWindow("imgs",cv2.NORM_MINMAX)
            cv2.imshow("imgs",imgs)

            key = cv2.waitKey()
            if key==27:
                exit()