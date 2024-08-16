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

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.anchor_target import AnchorTarget
from pysot.core.config import cfg
import random
import math
from pysot.utils.bbox import corner2center, \
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

class OneImgSeg(object):
    def __init__(self, img_path, msk_root,mask_json,interx,intery):
        self.img_path = img_path
        self.mask_json = mask_json
        self.msk_root = msk_root
        self.interx = interx
        self.intery = intery
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
        delta_weight = np.zeros((1, size, size), dtype=np.float32)
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
            if not cfg.DATASET.PERSONTK.FLAG_POINTTARGET_NEGALL:
                neg = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                               np.square(tcy - points[1]) / np.square(th / 4) < 1)
                neg, neg_num = select(neg, cfg.TRAIN.NEG_NUM)
                cls[neg] = 0
            else:
                cls[:] = 0

            return cls, delta,delta_weight

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
        if not cfg.DATASET.PERSONTK.FLAG_POINTTARGET_NEGALL:
            pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
            neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)
        else:
            pos_num = len(pos[0])
        delta_weight[0,pos] = 1. / (pos_num + 1e-6)
        cls[pos] = 1
        cls[neg] = 0

        return cls, delta,delta_weight


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
            return cls, delta, delta_weight

        delta[0] = points[0] - target[0]
        delta[1] = points[1] - target[1]
        delta[2] = target[2] - points[0]
        delta[3] = target[3] - points[1]

        # ellipse label
        pos = np.where(np.square(tcx - points[0]) / np.square(tw / cfg.POINT.SCALE_POS) +
                       np.square(tcy - points[1]) / np.square(th / cfg.POINT.SCALE_POS) < 1)
        pos_num = len(pos[0])
        delta_weight[0, pos] = 1. / (pos_num + 1e-6)
        cls[pos] = 1

        return cls, delta, delta_weight

class PointTarget2:
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
            return cls, delta, delta_weight

        delta[0] = points[0] - target[0]
        delta[1] = points[1] - target[1]
        delta[2] = target[2] - points[0]
        delta[3] = target[3] - points[1]

        # ellipse label
        mins = (tw + th)/2
        pos = np.where(np.square(tcx - points[0]) /np.square(mins/ cfg.POINT.SCALE_POS) + \
              np.square(tcy - points[1]) / np.square(mins/ cfg.POINT.SCALE_POS) < 1)
        pos_num = len(pos[0])
        delta_weight[0, pos] = 1. / (pos_num + 1e-6)
        cls[pos] = 1

        return cls, delta, delta_weight
def calciou(box1,box2):
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
    area1 = (x2_1 - x1_1)*(y2_1 - y1_1)
    area2 = (x2_2 - x1_2)*(y2_2 - y1_2)
    return float(area_inter)/float(area1 + area2 - area_inter + 1e-8),interx,intery

class TrkDataset(Dataset):
    def __init__(self,flag_alldatalist=True,world_size=None,batchsize=None):
        super(TrkDataset, self).__init__()
        # create anchor target
        self.anchor_target = AnchorTarget()
        """"
        Load Image Data
        """
        rootid = 0
        self.datasetroot_list = []
        self.meta_list_img = []
        self.meta_list_img_mult = []
        for name in cfg.DATASET.PERSONTK.IMAGE_NAMES:
            subdata_cfg = getattr(cfg.DATASET.PERSONTK, name)
            root = subdata_cfg.ROOT
            anno = subdata_cfg.ANNO
            num_use = subdata_cfg.NUM_USE
            self.datasetroot_list.append(root)
            data=json.load(open(anno))
            print(anno)
            cntins = 0
            for imgpath in data:
                boxes = data[imgpath]
                numbox = len(boxes)
                if numbox > 0:
                    if numbox in [1, 2]:
                        if numbox == 1:
                            self.meta_list_img.extend([[rootid, imgpath, boxes], ]*num_use)
                        else:
                            iou, interx, intery = calciou(boxes[0], boxes[1])
                            if iou == 0:
                                self.meta_list_img.extend([[rootid, imgpath, boxes, interx, intery],]*num_use)
                            else:
                                self.meta_list_img_mult.extend([[rootid, imgpath, boxes], ]*num_use)
                    else:
                        self.meta_list_img_mult.extend([[rootid, imgpath, boxes], ]*num_use)
            logger.info("Len of meta_list_img: %d; after loading %s;(numuse:%d)"%(len(self.meta_list_img),name,num_use))
            logger.info("Len of meta_list_img_mult: %d; after loading %s;(numuse:%d)"%(len(self.meta_list_img_mult),name,num_use))
            rootid += 1
        """"
                Load Video Data
                """
        self.meta_list_vid = []
        for name in cfg.DATASET.PERSONTK.VIDEO_NAMES:
            subdata_cfg = getattr(cfg.DATASET.PERSONTK, name)
            root = subdata_cfg.ROOT
            anno = subdata_cfg.ANNO
            num_use = subdata_cfg.NUM_USE
            self.datasetroot_list.append(root)
            data = json.load(open(anno))
            meta_list = [[0, rootid, m] for m in data]
            for i in range(num_use):
                self.meta_list_vid.extend(meta_list)
            rootid += 1
            logger.info("Len of meta_list_vid: %d; after loading %s;(numuse:%d)"%(len(self.meta_list_vid),name,num_use))


        """
        Add reid data
        """
        if ipaddress_int == 122:
            file_name = "/home/zhangming/REIDDATA/REID/Rcom3/partitions.pkl"
            imgroot_reid = "/home/zhangming/REIDDATA/REID/Combined_Duke_iLIDS_Market_MARS_REMOCollect_REID_20180704_V0/images"
        elif ipaddress_int in [125, 127]:
            file_name = "/home/inspur/Datasets/REID/Rcom3/partitions.pkl"
            imgroot_reid = "/home/inspur/SSD_DATA/REID/Combined_Duke_iLIDS_Market_MARS_REMOCollect_REID_20180704_V0/images"
        elif ipaddress_int in [128, ]:
            file_name = "/mnt/DataHDD/Datasets/REID/Rcom3/partitions.pkl"
            imgroot_reid = "/home/inspur/SSD_DATA/REID/Combined_Duke_iLIDS_Market_MARS_REMOCollect_REID_20180704_V0/images"
        self.datasetroot_list.append(imgroot_reid)
        print("reiddata:", file_name)
        data = load_pickle(file_name)
        im_dicts = {}
        for im_name in data["trainval_im_names"]:
            parsed = int(osp.basename(im_name)[:8])
            try:
                im_dicts[parsed].append(im_name)
            except:
                im_dicts[parsed] = [im_name, ]
        for key in im_dicts:
            self.meta_list_vid.append([1, rootid,im_dicts[key]])
        logger.info("Len of meta_list_vid: %d; after loading REID data"%(len(self.meta_list_vid)))

        random.shuffle(self.meta_list_img)
        random.shuffle(self.meta_list_img_mult)
        random.shuffle(self.meta_list_vid)

        """
        Add Person Segmentation Data
        """
        if hostname == "ZM116":
            dsetroot = "/home/zhangming/12f623b3-e657-49ea-a3bf-cb09ec1995ad/Datasets/Segmentation"
        elif hostname == "ZM113":
            dsetroot = "/home/zhangming/f87ff691-1e63-43a2-834c-ca90e9356c5f/Datasets/Segmentation"
        elif hostname == "REMO110":
            dsetroot = "/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/Segmentation"
        elif hostname == "REMO111":
            dsetroot = "/home/zhangming/47557749-05d8-4626-b22f-b1c43c475c5c/Datasets/Segmentation"
        elif hostname == "REMOTE022":
            dsetroot = "/home/zhangming/DatasetBack/Segmentation"
        elif hostname == "inspur-NF5468A5":
            if ipaddress_int == 126:
                dsetroot = "/mnt/data/Datasets/Segmentation"
            else:
                dsetroot = "/home/inspur/SSD_DATA/Segmentation"

        else:
            dsetroot = "/home/zhangming/SSD_DATA/Segmentation"
        print(dsetroot)
        TRAINSEG_JSON = [
            '/home/zhangming/SSD_DATA/Segmentation/seg_coco_LIP/seg_LIP_COCO_instance/seg_LIP_COCO_instance_dic_train.json',
            '/home/zhangming/SSD_DATA/Segmentation/seg_New_instance/seg_New_instance_allright/seg_New_instance_allright.json',
            '/home/zhangming/SSD_DATA/Segmentation/seg_New_instance/seg_New_instance_decorate/seg_New_instance_decorate.json',
        ]
        TRAINSEG_MASKROOT = [
            '/home/zhangming/SSD_DATA/Segmentation/seg_coco_LIP/seg_LIP_COCO_instance/LIP_COCO_InUse',
            '/home/zhangming/SSD_DATA/Segmentation/seg_New_instance/seg_New_instance_allright/seg_New_instance_mask',
            '/home/zhangming/SSD_DATA/Segmentation/seg_New_instance/seg_New_instance_decorate/seg_New_instance_mask',
        ]

        TRAINSEG_PICROOT = [
            '/home/zhangming/SSD_DATA/Segmentation/seg_coco_LIP/coco_LIP_mosaic_img',
            '/home/zhangming/SSD_DATA/Segmentation/seg_New_instance/seg_New_instance_allright/seg_New_instance_mosaic_img',
            '/home/zhangming/SSD_DATA/Segmentation/seg_New_instance/seg_New_instance_decorate/seg_New_instance_mosaic_img',
        ]

        jsonfile_instance = os.path.join(dsetroot,
                                         "REMOSeg_INSTANCE_FaceboxesLE2_Code018_02_20220125_FilterWithMask.json")
        """
        load instance data
        """
        def filterbox(maskinfo):
            mask_new = []
            for maskname, box in maskinfo:
                xmin, ymin, xmax, ymax = box
                if ymax > ymin and xmax > xmin and xmin >= 0 and ymin >= 0 and xmax <= 1 and ymax <= 1:
                    mask_new.append([maskname, box])
            return mask_new

        strsplit = "SSD_DATA/Segmentation"
        cntins = 0
        data_list = []
        data_list_mult = []

        for idx, maskjson_file in enumerate(TRAINSEG_JSON):
            pic_root = TRAINSEG_PICROOT[idx]
            msk_root = TRAINSEG_MASKROOT[idx]
            pic_root = pic_root.replace("/home/zhangming/SSD_DATA/Segmentation", dsetroot)
            msk_root = msk_root.replace("/home/zhangming/SSD_DATA/Segmentation", dsetroot)
            f1 = maskjson_file.replace("/home/zhangming/SSD_DATA/Segmentation", dsetroot)
            maskjson = json.load(open(f1))
            for picname in maskjson:
                imgpath = os.path.join(pic_root, picname)
                masknew = filterbox(maskjson[picname])
                if len(masknew) > 0 and os.path.exists(imgpath):
                    numins = len(masknew)
                    if numins in [1,2]:
                        if numins==1:
                            data_list.append(OneImgSeg(imgpath, msk_root, masknew,0,0))
                            cntins += 1
                        else:
                            boxa = masknew[0][1]
                            boxb = masknew[1][1]
                            iou, interx, intery = calciou(boxa, boxb)
                            if iou==0:
                                data_list.append(OneImgSeg(imgpath, msk_root, masknew, interx, intery))
                                cntins += 2
                    else:
                        data_list_mult.append(OneImgSeg(imgpath, msk_root, masknew,0,0))
                else:
                    print(imgpath)
            logger.info("Len of data_list seg: %d; after loading %s. "%(len(data_list),f1))
            logger.info("Len of data_list_mult seg: %d; after loading %s. "%(len(data_list_mult),f1))

        pic_root = os.path.join(dsetroot, 'seg_coco_LIP/coco_LIP_mosaic_img')
        msk_root = os.path.join(dsetroot, "seg_LIP_COCO_Extra_instance/LIP_COCO_EXTRA")
        jsonmask = "seg_LIP_COCO_Extra_instance_dic_train_faceclean.json"

        f1 = os.path.join(dsetroot, 'seg_LIP_COCO_Extra_instance', jsonmask)
        maskjson = json.load(open(f1))
        for picname in maskjson:
            imgpath = os.path.join(pic_root, picname)
            masknew = filterbox(maskjson[picname])
            if len(masknew) > 0 and os.path.exists(imgpath):
                numins = len(masknew)
                if numins in [1, 2]:
                    if numins == 1:
                        data_list.append(OneImgSeg(imgpath, msk_root, masknew, 0, 0))
                        cntins += 1
                    else:
                        boxa = masknew[0][1]
                        boxb = masknew[1][1]
                        iou, interx, intery = calciou(boxa, boxb)
                        if iou == 0:
                            data_list.append(OneImgSeg(imgpath, msk_root, masknew, interx, intery))
                            cntins += 2
                else:
                    data_list_mult.append(OneImgSeg(imgpath, msk_root, masknew, 0, 0))

            else:
                print(imgpath)
        logger.info("Len of data_list seg: %d; after loading %s. " % (len(data_list), f1))
        logger.info("Len of data_list_mult seg: %d; after loading %s. " % (len(data_list_mult), f1))

        self.meta_list_img_seg = data_list #55435
        self.meta_list_img_seg_mult = data_list_mult #55435

        videos_per_epoch = cfg.DATASET.PERSONTK.VIDEOS_PER_EPOCH
        self.flag_alldatalist = flag_alldatalist
        self.num = videos_per_epoch
        self.num *= cfg.TRAIN.EPOCH
        if world_size is not None and batchsize is not None:
            if self.num%(batchsize * world_size)!=0:
                n = self.num  // (batchsize * world_size)
                self.num = (n + 1) * (batchsize * world_size)
        """"
        add other object in got data
        """
        self.clstagforperson = ['1','n00007846','person']#coco,imagenetdet,ytb
        root_list_other = []
        meta_list_other = []
        cnt = -1
        for name in cfg.DATASET.NAMES:
            cnt += 1
            subdata_cfg = getattr(cfg.DATASET, name)
            root = subdata_cfg.ROOT
            root_list_other.append(root)
            jsonfile = subdata_cfg.ANNO
            with open(jsonfile, 'r') as f:
                meta_data = json.load(f)
                meta_data = self._filter_zero(meta_data)
            numvideo = len(meta_data.keys())
            cntvideo = 0
            cnttk = 0
            cnttkleft = 0
            for video in list(meta_data.keys()):
                tklist = list(meta_data[video].keys())
                cnttk += len(tklist)
                flagdel = False
                for track in tklist:
                    cls = str(meta_data[video][track]['cls'])
                    if cls in self.clstagforperson:
                        flagdel=True
                        break
                if not flagdel:
                    cntvideo += 1
                    for track in tklist:
                        frames = meta_data[video][track]
                        frames = list(map(int, filter(lambda x: x.isdigit(), frames.keys())))
                        frames.sort()
                        if len(frames)>0:
                            cnttkleft += 1
                            meta_data[video][track]['frames'] = frames
                        else:
                            del meta_data[video][track]
                else:
                    del meta_data[video]
            logger.warning("{} {} videos {} left; {} tracks {} left)".format(name, numvideo,cntvideo,cnttk,cnttkleft))
            cntvideo = len(meta_data.keys())
            cntdel = 0
            for video in list(meta_data.keys()):
                if len(meta_data[video]) <= 0:
                    # logger.warning("{} has no tracks".format(video))
                    del meta_data[video]
                    cntdel += 1
            logger.warning("{} delete {} videos({})".format(name, cntdel,cntvideo))

            meta_list = []
            for key in meta_data:
                meta_list.append([cnt,key,meta_data[key]])
            meta_list_other.extend(meta_list)
        self.meta_list_other = meta_list_other
        self.root_list_other = root_list_other

        self.shift_motion_model = True
        self.ScaleFactor = 1.0
        assert self.ScaleFactor==1
        self.mean_value_ = [117, 117, 117]
        if cfg.DATASET.PERSONTK.FLAG_TEMPLATESEARCH_SIZEFROMCONFIG:
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
        self.lambda_scale_ = cfg.DATASET.PERSONTK.LAMBDA_SCALE
        self.lambda_min_scale_ = cfg.DATASET.PERSONTK.LAMBDA_MIN_SCALE
        self.lambda_max_scale_ = cfg.DATASET.PERSONTK.LAMBDA_MAX_SCALE
        self.lambda_min_scale2_ = cfg.DATASET.PERSONTK.LAMBDA_MIN_SCALE2
        self.lambda_max_scale2_ = cfg.DATASET.PERSONTK.LAMBDA_MAX_SCALE2
        self.lambda_min_scale3_ = cfg.DATASET.PERSONTK.LAMBDA_MIN_SCALE3
        self.lambda_max_scale3_ = cfg.DATASET.PERSONTK.LAMBDA_MAX_SCALE3
        self.lambda_min_ratio_ = cfg.DATASET.PERSONTK.LAMBDA_MIN_RATIO
        self.lambda_max_ratio_ = cfg.DATASET.PERSONTK.LAMBDA_MAX_RATIO
        self.kContextFactorShiftBox = cfg.DATASET.PERSONTK.KCONTEXTFACTORSHIFTBOX
        self.anchor_target = AnchorTarget()
        self.num_img = len(self.meta_list_img)
        self.num_vid = len(self.meta_list_vid)
        self.num_img_seg = len(self.meta_list_img_seg)
        self.cnt_local = 0
        self.inprob = cfg.DATASET.PERSONTK.INPROB

        self.path_format = '{}.{}.{}.jpg'
        self.mask_format = "{}.{}.m.png"

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

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if frm.isdigit():
                        if not isinstance(bbox, dict):
                            if len(bbox) == 4:
                                x1, y1, x2, y2 = bbox
                                w, h = x2 - x1, y2 - y1
                            else:
                                w, h = bbox
                            if w <= 0 or h <= 0:
                                continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new
    def get_otherobj(self):
        rootid,video_name,video = random.choice(self.meta_list_other)
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frameid = random.choice(frames)
        frame = "{:06d}".format(frameid)
        image_path = os.path.join(self.root_list_other[rootid], video_name,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = track_info[frame]
        image = cv2.imread(image_path)
        box = self._get_bbox(image,image_anno)
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
        scale = float(self.resizedw_search)/float(self.resizedw_temp)*cfg.DATASET.PERSONTK.KCONTEXTFACTOR
        wa = float(wback/scale)
        ha = float(hback/scale)
        wa *= ((random.uniform(0, 1.0) - 0.5) * 2 * aspect_scale + 1.0)
        ha *= ((random.uniform(0, 1.0) - 0.5) * 2 * aspect_scale + 1.0)
        wa, ha = map(int, [wa, ha])

        xa_1 = random.randint(0, wback - wa - 1)
        ya_1 = random.randint(0, hback - ha - 1)
        xa_2 = xa_1 + wa
        ya_2 = ya_1 + ha
        xa_1, ya_1, xa_2, ya_2 = map(int,[xa_1, ya_1, xa_2, ya_2])

        imgper = cv2.resize(imgper, (wa, ha))
        imgback[ya_1:ya_2,xa_1:xa_2,:] = imgper
        xa_1 = float(xa_1)/self.resizedh_search
        ya_1 = float(ya_1)/self.resizedh_search
        xa_2 = float(xa_2)/self.resizedh_search
        ya_2 = float(ya_2)/self.resizedh_search

        return imgback, [xa_1, ya_1, xa_2, ya_2]

    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(3, 17, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size / 2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1 - wx)
            return kernel

        kernel = rand_kernel()
        image = cv2.filter2D(image, -1, kernel)
        return image

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        imgs_curr_all = []
        boxes_gt_all = []
        label_posnegpair = []

        seggt_list = []
        segzero = np.zeros((cfg.DATASET.PERSONTK.SEGGT_SIZE, cfg.DATASET.PERSONTK.SEGGT_SIZE),
                               dtype=np.uint8)
        segmask_weight = []
        flagrankloss = []
        ###step1:get negative pairs of other object
        for i in range(cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT):
            imageother, box_other = self.get_otherobj()
            I_currs, box_gts,_ = self.MakeTrainingExamples(1, imageother, box_other)
            imgs_curr_all.append(I_currs[0])
            boxes_gt_all.append(box_gts[0])
            label_posnegpair.append(0)
            seggt_list.append(segzero)
            segmask_weight.append(1)
            flagrankloss.append(1)

        flagimgdata = False
        if random.uniform(0,1.0)<cfg.DATASET.PERSONTK.PROB_IMG:
            flagimgdata = True
            ###step2: get positive pairs of imagedata
            # idroot,meta = self.meta_list_img[idximg]
            if random.uniform(0,1.0)<cfg.DATASET.PERSONTK.PROB_IMG_MULTIPLEIMAGE:
                meta = random.choice(self.meta_list_img_mult)
                rootid = meta[0]
                imgpath = meta[1]
                boxes = meta[2]
                img_prev = cv2.imread(os.path.join(self.datasetroot_list[rootid], imgpath))
                imgh, imgw = img_prev.shape[:2]
                bbox_prev = random.choice(boxes)
            else:
                meta = random.choice(self.meta_list_img)
                rootid = meta[0]
                imgpath = meta[1]
                boxes = meta[2]
                img_prev = cv2.imread(os.path.join(self.datasetroot_list[rootid], imgpath))
                imgh,imgw = img_prev.shape[:2]
                if len(boxes)==1:
                    bbox_prev = boxes[0]
                else:
                    interx, intery = meta[3:]
                    xmin1, ymin1, xmax1, ymax1 = boxes[0]
                    xmin2, ymin2, xmax2, ymax2 = boxes[1]
                    if interx == 0:
                        x2 = max(xmin1, xmin2)
                        cpbox1 = [0,0,x2,imgh]
                        x1 = min(xmax1,xmax2)
                        cpbox2 = [x1,0,imgw,imgh]
                    else:
                        y2 = max(ymin1, ymin2)
                        cpbox1 = [0,0,imgw,y2]
                        y1 = min(ymax1, ymax2)
                        cpbox2 = [0, y1, imgw, imgh]
                    cpbox = random.choice([cpbox1,cpbox2])
                    cpxmin,cpymin,cpxmax,cpymax = cpbox
                    if xmin1>= cpxmin and xmin1<=cpxmax and ymin1>=cpymin and ymin1<=cpymax and xmax1>= cpxmin and xmax1<=cpxmax and ymax1>=cpymin and ymax1<=cpymax:
                        xmin,ymin,xmax,ymax = xmin1, ymin1, xmax1, ymax1
                    else:
                        xmin, ymin, xmax, ymax = xmin2, ymin2, xmax2, ymax2
                    bbox_prev = [xmin - cpxmin,ymin - cpymin,xmax - cpxmin,ymax - cpymin]
                    img_prev = img_prev[cpymin:cpymax,cpxmin:cpxmax]

            img_curr = img_prev.copy()
            bbox_curr = bbox_prev[:]
            I_prev, imgs_curr, bboxes_gt,_ = self.MakeExamples(cfg.DATASET.PERSONTK.NUM_GENERATE_PER_IMAGE,
                                                                img_prev, img_curr, bbox_prev, bbox_curr)
            imgs_curr_all.extend(imgs_curr)
            boxes_gt_all.extend(bboxes_gt)
            label_posnegpair.extend([1]*len(imgs_curr))
            seggt_list.extend([segzero]*len(imgs_curr))
            segmask_weight.extend([0]*len(imgs_curr))
            flagrankloss.extend([1]*len(imgs_curr))
            nframepos = cfg.DATASET.PERSONTK.NUM_GENERATE_PER_IMAGE
        else:
            ###step4:get positive pairs of videodata
            # idset,vid_seq = self.meta_list_vid[idxvid]
            idset, rootid,vid_seq = random.choice(self.meta_list_vid)
            frame_id_prev = random.randint(0, len(vid_seq) - 1)
            frame_id_curr = random.randint(0, len(vid_seq) - 1)
            if idset == 0:  # video data
                imgpath_prev, bbox_prev = vid_seq[frame_id_prev]
                imgpath_curr, bbox_curr = vid_seq[frame_id_curr]
                img_prev = cv2.imread(os.path.join(self.datasetroot_list[rootid], imgpath_prev))
                img_curr = cv2.imread(os.path.join(self.datasetroot_list[rootid], imgpath_curr))
                I_prev, imgs_curr, bboxes_gt, _ = self.MakeExamples(cfg.DATASET.PERSONTK.NUM_GENERATE_PER_FRAME,
                                                  img_prev, img_curr, bbox_prev, bbox_curr)
                flagrankloss.extend([0] * len(imgs_curr))

            else:  # reid data
                img_prev = cv2.imread(os.path.join(self.datasetroot_list[rootid], os.path.basename(vid_seq[frame_id_prev])))
                img_curr = cv2.imread(os.path.join(self.datasetroot_list[rootid], os.path.basename(vid_seq[frame_id_curr])))
                imageback = random.choice(imgs_curr_all[:cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT]).copy()
                I_prev,bbox_prev = self.putimg_on_template(imageback.copy(), img_prev)
                bboxes_gt  = []
                imgs_curr = []
                for i in range(cfg.DATASET.PERSONTK.NUM_GENERATE_PER_FRAME):
                    img_c,bbox_curr = self.putimg_on_search(imageback.copy(), img_curr.copy())
                    imgs_curr.append(img_c)
                    bboxes_gt.append(bbox_curr)
                flagrankloss.extend([1] * len(imgs_curr))
            imgs_curr_all.extend(imgs_curr)
            boxes_gt_all.extend(bboxes_gt)
            label_posnegpair.extend([1]*len(imgs_curr))
            seggt_list.extend([segzero] * len(imgs_curr))
            segmask_weight.extend([0] *len(imgs_curr))
            nframepos = cfg.DATASET.PERSONTK.NUM_GENERATE_PER_FRAME


        ###step7: add segmentation data
        for i in range(cfg.DATASET.PERSONTK.NUM_FEATCH_SEG):
            if random.uniform(0,1.0)<cfg.DATASET.PERSONTK.PROB_IMG_MULTIPLEIMAGE:
                meta = random.choice(self.meta_list_img_seg_mult)
                img_prev = meta.get_img()
                segmentation,bbox_prev = meta.get_random_person()
                imgh, imgw = img_prev.shape[:2]
                x1, y1, x2, y2 = bbox_prev
                x1 *= imgw
                y1 *= imgh
                x2 *= imgw
                y2 *= imgh
                bbox_prev = [x1, y1, x2, y2]
            else:
                idseg = (index + i) % self.num_img_seg
                meta = self.meta_list_img_seg[idseg]
                img_prev = meta.get_img()
                mask_json = meta.mask_json
                imgh, imgw = img_prev.shape[:2]

                if len(mask_json)==1:
                    maskname,bbox_prev = mask_json[0]
                    segmentation = cv2.imread(os.path.join(meta.msk_root, maskname), flags=cv2.IMREAD_GRAYSCALE)
                    x1, y1, x2, y2 = bbox_prev
                    x1 *= imgw
                    y1 *= imgh
                    x2 *= imgw
                    y2 *= imgh
                    bbox_prev = [x1, y1, x2, y2]

                else:
                    interx, intery = meta.interx, meta.intery
                    xmin1, ymin1, xmax1, ymax1 = mask_json[0][1]
                    xmin2, ymin2, xmax2, ymax2 = mask_json[1][1]
                    xmin1 *= imgw
                    ymin1 *= imgh
                    xmax1 *= imgw
                    ymax1 *= imgh
                    xmin2 *= imgw
                    ymin2 *= imgh
                    xmax2 *= imgw
                    ymax2 *= imgh
                    xmin1, ymin1, xmax1, ymax1 = map(int,[xmin1, ymin1, xmax1, ymax1])
                    xmin2, ymin2, xmax2, ymax2 = map(int,[xmin2, ymin2, xmax2, ymax2])

                    if interx == 0:
                        x2 = max(xmin1, xmin2)
                        cpbox1 = [0, 0, x2, imgh]
                        x1 = min(xmax1, xmax2)
                        cpbox2 = [x1, 0, imgw, imgh]
                    else:
                        y2 = max(ymin1, ymin2)
                        cpbox1 = [0, 0, imgw, y2]
                        y1 = min(ymax1, ymax2)
                        cpbox2 = [0, y1, imgw, imgh]

                    cpbox = random.choice([cpbox1,cpbox2])
                    cpxmin, cpymin, cpxmax, cpymax = cpbox
                    if xmin1 >= cpxmin and xmin1 <= cpxmax and ymin1 >= cpymin and ymin1 <= cpymax and xmax1 >= cpxmin and xmax1 <= cpxmax and ymax1 >= cpymin and ymax1 <= cpymax:
                        xmin, ymin, xmax, ymax = xmin1, ymin1, xmax1, ymax1
                        idx = 0
                    else:
                        xmin, ymin, xmax, ymax = xmin2, ymin2, xmax2, ymax2
                        idx = 1
                    bbox_prev = [xmin - cpxmin, ymin - cpymin, xmax - cpxmin, ymax - cpymin]
                    maskname, _ = mask_json[idx]
                    segmentation = cv2.imread(os.path.join(meta.msk_root, maskname), flags=cv2.IMREAD_GRAYSCALE)
                    cpxmin, cpymin, cpxmax, cpymax = map(int,[cpxmin, cpymin, cpxmax, cpymax])
                    img_prev = img_prev[cpymin:cpymax, cpxmin:cpxmax]
                    segmentation = segmentation[cpymin:cpymax, cpxmin:cpxmax]

                    imgh, imgw = img_prev.shape[:2]

            if max(imgw, imgh) > 1500:
                s = 1500.0 / max(imgw, imgh)
                img_prev = cv2.resize(img_prev, dsize=None, fx=s, fy=s)
                segmentation = cv2.resize(segmentation, dsize=None, fx=s, fy=s)
                xmin,ymin,xmax,ymax = bbox_prev
                xmin *= s
                ymin *= s
                xmax *= s
                ymax *= s
                bbox_prev = [xmin,ymin,xmax,ymax]


            imgs_curr, bboxes_gt, seg = self.MakeTrainingExamples(1, img_prev, bbox_prev, segmentation)
            imgs_curr_all.append(imgs_curr[0])
            boxes_gt_all.append(bboxes_gt[0])
            label_posnegpair.append(1)
            seg = cv2.resize(seg[0], (cfg.DATASET.PERSONTK.SEGGT_SIZE, cfg.DATASET.PERSONTK.SEGGT_SIZE))
            seggt_list.append(seg)
            segmask_weight.append(1)
            flagrankloss.append(1)


        I_prev = Distortion((I_prev))
        if cfg.DATASET.PERSONTK.FLIP_TEMPLATE and cfg.DATASET.PERSONTK.FLIP_TEMPLATE > np.random.random():
            I_prev = cv2.flip(I_prev, 1)
        if cfg.DATASET.PERSONTK.FLIPVER_TEMPLATE and cfg.DATASET.PERSONTK.FLIPVER_TEMPLATE > np.random.random():
            I_prev = cv2.flip(I_prev, 0)
        n = cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT+cfg.DATASET.PERSONTK.NUM_GENERATE_PER_IMAGE + cfg.DATASET.PERSONTK.NUM_GENERATE_PER_FRAME+cfg.DATASET.PERSONTK.NUM_FEATCH_SEG
        # imgs_prev_all.extend([I_prev] * n)

        ###############################finish get data

        for i in range(len(imgs_curr_all)):
            imgs_curr_all[i] = Distortion(imgs_curr_all[i])
        for i in range(cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT):
            if cfg.DATASET.PERSONTK.FLIP_SEARCH and cfg.DATASET.PERSONTK.FLIP_SEARCH > np.random.random():
                imgs_curr_all[i] = cv2.flip(imgs_curr_all[i], 1)
                x1, y1, x2, y2 = boxes_gt_all[i]
                x1_new = self.ScaleFactor - x2
                x2_new = self.ScaleFactor - x1
                boxes_gt_all[i] = [x1_new, y1, x2_new, y2]
            if cfg.DATASET.PERSONTK.FLIPVER_SEARCH and cfg.DATASET.PERSONTK.FLIPVER_SEARCH > np.random.random():
                imgs_curr_all[i] = cv2.flip(imgs_curr_all[i], 0)
                x1, y1, x2, y2 = boxes_gt_all[i]
                y1_new = self.ScaleFactor - y2
                y2_new = self.ScaleFactor - y1
                boxes_gt_all[i] = [x1, y1_new, x2, y2_new]

        if cfg.DATASET.PERSONTK.FLIP_SEARCH and cfg.DATASET.PERSONTK.FLIP_SEARCH > np.random.random():
            for i in range(cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT,
                           cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT + nframepos):
                imgs_curr_all[i] = cv2.flip(imgs_curr_all[i], 1)
                x1, y1, x2, y2 = boxes_gt_all[i]
                x1_new = self.ScaleFactor - x2
                x2_new = self.ScaleFactor - x1
                boxes_gt_all[i] = [x1_new, y1, x2_new, y2]
        if cfg.DATASET.PERSONTK.FLIPVER_SEARCH and cfg.DATASET.PERSONTK.FLIPVER_SEARCH > np.random.random():
            for i in range(cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT,
                           cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT + nframepos):
                imgs_curr_all[i] = cv2.flip(imgs_curr_all[i], 0)
                x1, y1, x2, y2 = boxes_gt_all[i]
                y1_new = self.ScaleFactor - y2
                y2_new = self.ScaleFactor - y1
                boxes_gt_all[i] = [x1, y1_new, x2, y2_new]
        if cfg.DATASET.PERSONTK.ROTATE_SEARCH and cfg.DATASET.PERSONTK.ROTATE_SEARCH > np.random.random():
            if random.uniform(0.0, 1.0) < 0.5:
                for i in range(cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT,
                               cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT + nframepos):
                    imgs_curr_all[i] = cv2.rotate(imgs_curr_all[i], cv2.ROTATE_90_CLOCKWISE)
                    x1, y1, x2, y2 = boxes_gt_all[i]
                    x1_new = self.ScaleFactor - y2
                    y1_new = x1
                    x2_new = self.ScaleFactor - y1
                    y2_new = x2
                    boxes_gt_all[i] = [x1_new, y1_new, x2_new, y2_new]
            else:
                for i in range(cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT,
                               cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT + nframepos):
                    imgs_curr_all[i] = cv2.rotate(imgs_curr_all[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
                    x1, y1, x2, y2 = boxes_gt_all[i]
                    x1_new = y1
                    y1_new = self.ScaleFactor - x2
                    x2_new = y2
                    y2_new = self.ScaleFactor - x1
                    boxes_gt_all[i] = [x1_new, y1_new, x2_new, y2_new]



        for i in range(cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT + nframepos,
                       cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT + nframepos + cfg.DATASET.PERSONTK.NUM_FEATCH_SEG):  # only erease segmentation
            x1, y1, x2, y2 = boxes_gt_all[i]
            if y2 <= 1.0:
                if random.uniform(0, 1.0) < cfg.DATASET.PERSONTK.PROB_EREASE_BOTTOM:
                    xmin = x1 * self.resizedw_search
                    xmax = x2 * self.resizedw_search
                    ymin = y1 * self.resizedh_search
                    ymax = y2 * self.resizedh_search

                    xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                    h = ymax - ymin
                    scale = random.uniform(0.1, cfg.DATASET.PERSONTK.SCALE_EREASE_BOTTOM)
                    y1c = int(ymax - h * scale)
                    y2c = ymax
                    x1c = max(xmin, 0)
                    x2c = min(xmax, self.resizedh_search)
                    wc = x2c - x1c
                    hc = y2c - y1c
                    if wc > 0 and hc > 0 and x1c > 0 and y1c > 0 and x2c < self.resizedh_search and y2c < self.resizedh_search:
                        p = np.random.randint(0, 255, (hc, wc, 3))
                        imgs_curr_all[i][y1c:y2c, x1c:x2c, :] = p
                        y2 = float(y1c) / self.resizedh_search
                        x1c = int(float(x1c) / self.resizedh_search * cfg.DATASET.PERSONTK.SEGGT_SIZE)
                        y1c = int(float(y1c) / self.resizedh_search * cfg.DATASET.PERSONTK.SEGGT_SIZE)
                        x2c = int(float(x2c) / self.resizedh_search * cfg.DATASET.PERSONTK.SEGGT_SIZE)
                        y2c = int(float(y2c) / self.resizedh_search * cfg.DATASET.PERSONTK.SEGGT_SIZE)
                        if y2c > y1c and x2c > x1c:
                            seggt_list[i][y1c:y2c, x1c:x2c] = 0
                            ys, xs = np.where(seggt_list[i] > 0)
                            if len(ys) > 0:
                                xmin, ymin, xmax, ymax = xs.min(), ys.min(), xs.max(), ys.max()
                                xmin = float(xmin) / cfg.DATASET.PERSONTK.SEGGT_SIZE
                                ymin = float(ymin) / cfg.DATASET.PERSONTK.SEGGT_SIZE
                                xmax = float(xmax) / cfg.DATASET.PERSONTK.SEGGT_SIZE
                                ymax = float(ymax) / cfg.DATASET.PERSONTK.SEGGT_SIZE
                                x1 = max(x1, xmin)
                                y1 = max(y1, ymin)
                                x2 = min(x2, xmax)
                                y2 = min(y2, ymax)
                        boxes_gt_all[i] = [x1, y1, x2, y2]

        for i in range(cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT + nframepos,
                       cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT + nframepos + cfg.DATASET.PERSONTK.NUM_FEATCH_SEG):
            if cfg.DATASET.PERSONTK.FLIP_SEARCH and cfg.DATASET.PERSONTK.FLIP_SEARCH > np.random.random():
                imgs_curr_all[i] = cv2.flip(imgs_curr_all[i], 1)
                x1, y1, x2, y2 = boxes_gt_all[i]
                x1_new = self.ScaleFactor - x2
                x2_new = self.ScaleFactor - x1
                boxes_gt_all[i] = [x1_new, y1, x2_new, y2]
                if segmask_weight[i] == 1:
                    seggt_list[i] = cv2.flip(seggt_list[i], 1)
            if cfg.DATASET.PERSONTK.FLIPVER_SEARCH and cfg.DATASET.PERSONTK.FLIPVER_SEARCH > np.random.random():
                imgs_curr_all[i] = cv2.flip(imgs_curr_all[i], 0)
                x1, y1, x2, y2 = boxes_gt_all[i]
                y1_new = self.ScaleFactor - y2
                y2_new = self.ScaleFactor - y1
                boxes_gt_all[i] = [x1, y1_new, x2, y2_new]
                if segmask_weight[i] == 1:
                    seggt_list[i] = cv2.flip(seggt_list[i], 0)

            if cfg.DATASET.PERSONTK.ROTATE_SEARCH and cfg.DATASET.PERSONTK.ROTATE_SEARCH > np.random.random():
                if random.uniform(0.0, 1.0) < 0.5:
                    imgs_curr_all[i] = cv2.rotate(imgs_curr_all[i], cv2.ROTATE_90_CLOCKWISE)
                    x1, y1, x2, y2 = boxes_gt_all[i]
                    x1_new = self.ScaleFactor - y2
                    y1_new = x1
                    x2_new = self.ScaleFactor - y1
                    y2_new = x2
                    boxes_gt_all[i] = [x1_new, y1_new, x2_new, y2_new]
                    if segmask_weight[i] == 1:
                        seggt_list[i] = cv2.rotate(seggt_list[i], cv2.ROTATE_90_CLOCKWISE)
                else:
                    imgs_curr_all[i] = cv2.rotate(imgs_curr_all[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
                    x1, y1, x2, y2 = boxes_gt_all[i]
                    x1_new = y1
                    y1_new = self.ScaleFactor - x2
                    x2_new = y2
                    y2_new = self.ScaleFactor - x1
                    boxes_gt_all[i] = [x1_new, y1_new, x2_new, y2_new]
                    if segmask_weight[i] == 1:
                        seggt_list[i] = cv2.rotate(seggt_list[i], cv2.ROTATE_90_COUNTERCLOCKWISE)

        if flagimgdata:
            for i in range(cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT,
                           cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT + nframepos):
                if random.uniform(0.0,1.0) < cfg.DATASET.PERSONTK.PROB_IMAGEDATA_BLUR:
                    imgs_curr_all[i] = self._blur_aug(imgs_curr_all[i])
        for i in range(cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT + nframepos,
                       cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT + nframepos + cfg.DATASET.PERSONTK.NUM_FEATCH_SEG):
            if random.uniform(0.0, 1.0) < cfg.DATASET.PERSONTK.PROB_IMAGEDATA_BLUR:
                imgs_curr_all[i] = self._blur_aug(imgs_curr_all[i])

        for i in range(cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT,
                       cfg.DATASET.PERSONTK.NUM_NEG_OTHEROBJECT + nframepos+cfg.DATASET.PERSONTK.NUM_FEATCH_SEG):
            if random.uniform(0, 1.0)<cfg.DATASET.PERSONTK.PROB_MIRRORTOPBOT_AUG:
                x1, y1, x2, y2 = boxes_gt_all[i]
                xmin = x1 * self.resizedw_search
                xmax = x2 * self.resizedw_search
                ymin = y1 * self.resizedh_search
                ymax = y2 * self.resizedh_search
                xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                gap = min(ymax, self.resizedh_search - ymax)
                if ymax < self.resizedh_search and ymax > 0:
                    if cfg.DATASET.PERSONTK.MIRR_MODE == 0:
                        imgs_curr_all[i][ymax:ymax + gap, :, :] = imgs_curr_all[i][ymax:ymax - gap:-1, :, :]*0.4 + imgs_curr_all[i][ymax:ymax + gap, :, :]*0.6
                    elif cfg.DATASET.PERSONTK.MIRR_MODE == 1:
                        mirr_alpha = random.uniform(cfg.DATASET.PERSONTK.MIRR_ALPHA_MAX, cfg.DATASET.PERSONTK.MIRR_ALPHA_MIN)
                        imgs_curr_all[i][ymax:ymax + gap, :, :] = imgs_curr_all[i][ymax:ymax - gap:-1, :, :]*mirr_alpha + imgs_curr_all[i][ymax:ymax + gap, :, :]*(1-mirr_alpha)

        assert len(label_posnegpair)==len(boxes_gt_all)
        cls_list = []
        delta_list = []
        searchboxes = []
        delta_weight_list = []
        for i in range(len(label_posnegpair)):
            flagneg = label_posnegpair[i]==0
            x1,y1,x2,y2 = boxes_gt_all[i]
            x1 *= self.resizedw_search
            x2 *= self.resizedw_search
            y1 *= self.resizedh_search
            y2 *= self.resizedh_search
            bbox = Corner(x1, y1, x2, y2)
            searchboxes.append([x1,y1,x2,y2])
            cls, delta,delta_weight, overlap = self.anchor_target(bbox, cfg.TRAIN.OUTPUT_SIZE, flagneg)
            cls_list.append(cls)
            delta_list.append(delta)
            delta_weight_list.append(delta_weight)

        # template = np.stack(imgs_prev_all).transpose((0, 3, 1, 2))
        # print(type(template[0,0,0,0]))
        I_prev = I_prev.transpose((2, 0, 1))
        search = np.stack(imgs_curr_all).transpose((0, 3, 1, 2))
        cls = np.stack(cls_list)
        delta = np.stack(delta_list)
        delta_weight = np.stack(delta_weight_list)
        seggt_all = np.expand_dims(np.stack(seggt_list), axis=1)
        segmask_weight = np.array(segmask_weight).reshape(-1, 1)
        searchboxes = np.array(searchboxes)
        flagrankloss = np.array(flagrankloss)

        return {
            # 'template': template,
            'template': I_prev,
            'search': search,
            'label_cls': cls,
            'label_loc': delta,
            'label_loc_weight': delta_weight,
            'seggt_all': seggt_all,
            'segmask_weight': segmask_weight,
            'searchboxes':searchboxes,
            'label_posnegpair':label_posnegpair,
            'flagrankloss': flagrankloss

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
        if cfg.DATASET.PERSONTK.TYPE_CONTEXTBOX == 0:
            w = (x2 - x1) * cfg.DATASET.PERSONTK.KCONTEXTFACTOR
            h = (y2 - y1) * cfg.DATASET.PERSONTK.KCONTEXTFACTOR
        elif cfg.DATASET.PERSONTK.TYPE_CONTEXTBOX == 1:
            box_w = x2 - x1
            box_h = y2 - y1
            w = box_w + cfg.DATASET.PERSONTK.KCONTEXTFACTOR * (box_w + box_h)
            h = box_h + cfg.DATASET.PERSONTK.KCONTEXTFACTOR * (box_w + box_h)
        elif cfg.DATASET.PERSONTK.TYPE_CONTEXTBOX == 2:
            w = (x2 - x1) * cfg.DATASET.PERSONTK.KCONTEXTFACTOR
            h = (y2 - y1) * cfg.DATASET.PERSONTK.KCONTEXTFACTOR
            w = max(w, h)
            h = w
        elif cfg.DATASET.PERSONTK.TYPE_CONTEXTBOX == 3:
            box_w = x2 - x1
            box_h = y2 - y1
            w = box_w + cfg.DATASET.PERSONTK.KCONTEXTFACTOR * (box_w + box_h)
            h = box_h + cfg.DATASET.PERSONTK.KCONTEXTFACTOR * (box_w + box_h)
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
            if cfg.DATASET.PERSONTK.SHIFT_MODE == 2:
                crop_box = self.shift_bbox_remo2(bbox_curr)
            elif cfg.DATASET.PERSONTK.SHIFT_MODE == 3:
                crop_box = self.shift_bbox_remo3(bbox_curr)
            elif cfg.DATASET.PERSONTK.SHIFT_MODE == 4:
                crop_box = self.shift_bbox_remo4(bbox_curr)
            else:
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
        if cfg.DATASET.PERSONTK.TYPE_CONTEXTBOX == 0:
            w = (x2 - x1) * cfg.DATASET.PERSONTK.KCONTEXTFACTOR
            h = (y2 - y1) * cfg.DATASET.PERSONTK.KCONTEXTFACTOR
        elif cfg.DATASET.PERSONTK.TYPE_CONTEXTBOX == 1:
            box_w = x2 - x1
            box_h = y2 - y1
            w = box_w + cfg.DATASET.PERSONTK.KCONTEXTFACTOR * (box_w + box_h)
            h = box_h + cfg.DATASET.PERSONTK.KCONTEXTFACTOR * (box_w + box_h)
        elif cfg.DATASET.PERSONTK.TYPE_CONTEXTBOX == 2:
            w = (x2 - x1) * cfg.DATASET.PERSONTK.KCONTEXTFACTOR
            h = (y2 - y1) * cfg.DATASET.PERSONTK.KCONTEXTFACTOR
            w = max(w, h)
            h = w
        elif cfg.DATASET.PERSONTK.TYPE_CONTEXTBOX == 3:
            box_w = x2 - x1
            box_h = y2 - y1
            w = box_w + cfg.DATASET.PERSONTK.KCONTEXTFACTOR * (box_w + box_h)
            h = box_h + cfg.DATASET.PERSONTK.KCONTEXTFACTOR * (box_w + box_h)
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
                width_scale_factor = self.lambda_min_scale_ + rand_num(self.lambda_max_scale_ - self.lambda_min_scale_)
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

    def coveA(self,box1, box2):
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

    def shift_bbox_remo(self,box):  # box = [xmin,ymin,xmax,ymax]
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
            if self.coveA(box, [x1, y1, x2, y2]) > cfg.DATASET.PERSONTK.COVA_EMIT:
                flagok = True
                break
        if not flagok:
            x1 = cx_box - cp_w / 2
            y1 = cy_box - cp_h / 2
            x2 = x1 + cp_w
            y2 = y1 + cp_h
        return [x1, y1, x2, y2]

    def shift_bbox_remo2(self, box):  # box = [xmin,ymin,xmax,ymax]
        """

        self.lambda_min_scale_ = -0.4
        self.lambda_max_scale_ = 0.4
        """
        xmin, ymin, xmax, ymax = box
        cp_w = (xmax - xmin) * 2
        cp_h = (ymax - ymin) * 2
        cx_box = (xmin + xmax) / 2
        cy_box = (ymax + ymin) / 2

        s_rand = self.lambda_min_scale_ + random.uniform(0, 1.0) * (self.lambda_max_scale_ - self.lambda_min_scale_)
        r_rand = self.lambda_min_ratio_ + random.uniform(0, 1.0) * (self.lambda_max_ratio_ - self.lambda_min_ratio_)
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
            if self.coveA(box, [x1, y1, x2, y2]) > cfg.DATASET.PERSONTK.COVA_EMIT:
                flagok = True
                break
        if not flagok:
            x1 = cx_box - cp_w / 2
            y1 = cy_box - cp_h / 2
            x2 = x1 + cp_w
            y2 = y1 + cp_h
        return [x1, y1, x2, y2]
    def shift_bbox_remo3(self, box):  # box = [xmin,ymin,xmax,ymax]
        """

        self.lambda_min_scale_ = -0.4
        self.lambda_max_scale_ = 0.4
        """
        xmin, ymin, xmax, ymax = box
        cp_w = (xmax - xmin) * 2
        cp_h = (ymax - ymin) * 2
        cx_box = (xmin + xmax) / 2
        cy_box = (ymax + ymin) / 2


        if random.uniform(0, 1.0) < self.inprob:
            s_rand = self.lambda_min_scale_ + random.uniform(0, 1.0) * (self.lambda_max_scale_ - self.lambda_min_scale_)
        elif random.uniform(0, 1.0) < self.inprob:
            s_rand = self.lambda_min_scale2_ + random.uniform(0, 1.0) * (self.lambda_max_scale2_ - self.lambda_min_scale2_)
        else:
            s_rand = self.lambda_min_scale3_ + random.uniform(0, 1.0) * (self.lambda_max_scale3_ - self.lambda_min_scale3_)
        r_rand = self.lambda_min_ratio_ + random.uniform(0, 1.0) * (self.lambda_max_ratio_ - self.lambda_min_ratio_)

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
            if self.coveA(box, [x1, y1, x2, y2]) > cfg.DATASET.PERSONTK.COVA_EMIT:
                flagok = True
                break
        if not flagok:
            x1 = cx_box - cp_w / 2
            y1 = cy_box - cp_h / 2
            x2 = x1 + cp_w
            y2 = y1 + cp_h
        return [x1, y1, x2, y2]

    def shift_bbox_remo4(self, box):

        xmin, ymin, xmax, ymax = box
        cp_w = (xmax - xmin) * 2
        cp_h = (ymax - ymin) * 2
        cx_box = (xmin + xmax) / 2
        cy_box = (ymax + ymin) / 2
        r_rand = self.lambda_min_ratio_ + random.uniform(0, 1.0) * (self.lambda_max_ratio_ - self.lambda_min_ratio_)

        s_rand = math.log(random.random()) / self.lambda_scale_
        s_rand = max(self.lambda_min_scale_, min(self.lambda_max_scale_, s_rand))

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
            if self.coveA(box, [x1, y1, x2, y2]) > cfg.DATASET.PERSONTK.COVA_EMIT:
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

    # import socket
    #
    # hostname = socket.gethostname()
    #
    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # s.connect(("8.8.8.8", 80))
    # ipstr = s.getsockname()[0]
    # ipaddress_int = int(ipstr.split('.')[-1])
    # if ipaddress_int==111:
    #     cfg.DATASET.PERSONTK.DATASET_ROOT = '/home/zhangming/Datasets/GOT/PersonTracker'
    #     cfg.DATASET.PERSONTK.IMAGE_LIST = ('/home/zhangming/Datasets/GOT/PersonTracker/Layout/persontracker_imagedata_cocovoc_20220802.json',)
    #     cfg.DATASET.PERSONTK.VID_LIST = ('/home/zhangming/Datasets/GOT/PersonTracker/Layout/alov.json',
    #                                      '/home/zhangming/Datasets/GOT/PersonTracker/Layout/otb.json',
    #                                      '/home/zhangming/Datasets/GOT/PersonTracker/Layout/TrackerNet.json',
    #                                      '/home/zhangming/Datasets/GOT/PersonTracker/Layout/VOT2016.json',
    #                                      '/home/zhangming/Datasets/GOT/PersonTracker/Layout/Youtube.json',
    #                                      )
    # elif ipaddress_int==110:
    #     cfg.DATASET.PERSONTK.DATASET_ROOT = '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/PersonTracker'
    #     cfg.DATASET.PERSONTK.IMAGE_LIST = (
    #     '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/PersonTracker/Layout/persontracker_imagedata_cocovoc_20220802.json',)
    #     cfg.DATASET.PERSONTK.VID_LIST = ('/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/PersonTracker/Layout/alov.json',
    #                                      '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/PersonTracker/Layout/otb.json',
    #                                      '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/PersonTracker/Layout/TrackerNet.json',
    #                                      '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/PersonTracker/Layout/VOT2016.json',
    #                                      '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/PersonTracker/Layout/Youtube.json',
    #                                      )
    #     cfg.DATASET.NAMES = ('VID','YOUTUBEBB','COCO','DET')
    #
    #     cfg.DATASET.VID.ROOT = '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/ILSVRC2015/crop511'
    #     cfg.DATASET.VID.ANNO = '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/ILSVRC2015/train.json'
    #     cfg.DATASET.YOUTUBEBB.ROOT = '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/youtube/youtube_new'
    #     cfg.DATASET.YOUTUBEBB.ANNO = '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/youtube/train.json'
    #     cfg.DATASET.COCO.ROOT = '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/coco/crop511'
    #     cfg.DATASET.COCO.ANNO = '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/coco/train2017.json'
    #     cfg.DATASET.DET.ROOT = '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/ImageNetDet/crop511'
    #     cfg.DATASET.DET.ANNO = '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/ImageNetDet/train.json'
    #
    # cfg.DATASET.PERSONTK.KCONTEXTFACTOR = 2.0
    # cfg.DATASET.PERSONTK.TYPE_CONTEXTBOX = 0
    # cfg.DATASET.PERSONTK.VIDEOS_PER_EPOCH = 600000
    # cfg.DATASET.PERSONTK.CHANGEPADDING_VALUE = (104, 117, 123)
    # cfg.DATASET.PERSONTK.GRAY = 0.0
    # cfg.DATASET.PERSONTK.FETCH_ITERS = 2
    # cfg.TRAIN.SEARCH_SIZE = 160
    # cfg.TRAIN.OUTPUT_SIZE = 11
    # cfg.DATASET.FLAG_CHECKDESIRED_OUTPUT = False
    # cfg.ANCHOR.STRIDE = 8
    # cfg.ANCHOR.RATIOS = [0.5,1,2.0]
    # cfg.ANCHOR.SCALES = [11]
    # cfg.ANCHOR.ANCHOR_NUM = 3
    #
    # cfg.DATASET.PERSONTK.FLIP_TEMPLATE = 0.0
    # cfg.DATASET.PERSONTK.FLIPVER_TEMPLATE = 0.0
    # cfg.DATASET.PERSONTK.FLIP_SEARCH = 0.0
    # cfg.DATASET.PERSONTK.FLIPVER_SEARCH = 0.0

    cfgfile = "/home/inspur/work/pysot_REMO/experiments/siamrpn_pertk_darknet/20221128GenDetV009PerTKMobileOneWeightAdd_ACMOutPointMaskFromBB_DsetV6RandImgVidOneInsSFTREM_4GPU127.yaml"
    cfg.merge_from_file(cfgfile)
    cfg.DATASET.PERSONTK.DATASET_ROOT_OTHERIMAGE="/home/inspur/SSD_DATA/AIC_Data/"
    cfg.DATASET.PERSONTK.IMAGE_LIST=(
        '/home/inspur/Datasets/AIC_Data/keypoint_validation_annotations_20170911.json',
        '/home/inspur/Datasets/AIC_Data/keypoint_train_annotations_20170909.json'
    )

    # cfg.DATASET.NAMES=('COCO','VID','DET')
    # cfg.DATASET.VID.ANNO = '/home/inspur/SSD_DATA/GOT/ILSVRC2015/train_withclassname.json'
    # cfg.DATASET.COCO.ANNO = '/home/inspur/SSD_DATA/GOT/coco/train2017_withclassname.json'
    # cfg.DATASET.DET.ANNO = '/home/inspur/SSD_DATA/GOT/ImageNetDet/train_withclassname.json'
    # cfg.DATASET.PERSONTK.PROB_EREASE_BOTTOM = 1.0
    # cfg.DATASET.PERSONTK.FLAG_TEMPLATESEARCH_SIZEFROMCONFIG = False
    # cfg.TRAIN.EXEMPLAR_SIZE = 160
    # cfg.TRAIN.SEARCH_SIZE = 160
    # cfg.TRAIN.KCONTEXTFACTORSHIFTBOX = 2.0
    # cfg.TRAIN.EPOCH = 2
    """
    {
            # 'template': template,
            'template': I_prev,
            'search': search,
            'label_cls': cls,
            'label_loc': delta,
            'label_loc_weight': delta_weight,
            'seggt_all': seggt_all,
            'segmask_weight': segmask_weight,
            'searchboxes':searchboxes,
            'label_posnegpair':label_posnegpair

        }
    """
    data = TrkDataset()
    idlist = list(range(len(data)))
    random.shuffle(idlist)
    cnt = -1
    for i in idlist[:10]:
        a = data.__getitem__(i)
        I_prev =  a["template"]
        I_prev = I_prev.transpose((1,2,0))
        search = a["search"]
        seggt_all = a["seggt_all"]
        template = a['template']
        label_cls = a['label_cls']
        boxes_gt_all = a['searchboxes']
        label_posnegpair = a['label_posnegpair']
        nimg = search.shape[0]
        cnt += 1
        cv2.imwrite("images/%06d_t.jpg" % cnt, I_prev)
        for j in range(nimg):
            # imgs = search[-(j+1)].transpose((1,2,0))
            # imgt = template[-(j+1)].transpose((1,2,0))
            # loc = boxes_gt_all[-(j+1)]
            imgs = search[j].transpose((1, 2, 0))
            imgh,imgw = imgs.shape[:2]
            cls = label_cls[j]

            loc = boxes_gt_all[j]
            x1,y1,x2,y2 = loc
            x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
            if label_posnegpair[j]==1:
                cv2.rectangle(imgs,(x1,y1),(x2,y2),(0,0,255))
            else:
                cv2.rectangle(imgs,(x1,y1),(x2,y2),(0,255,0))

            imgseg = seggt_all[j].squeeze()*255
            imgseg = imgseg.astype(np.uint8)
            cv2.imwrite("images/%06d_%d_s.jpg"%(cnt,j),imgs)
            cv2.imwrite("images/%06d_%d_seg.jpg"%(cnt,j),imgseg)


            # cv2.namedWindow("imgs",cv2.NORM_MINMAX)
            # cv2.imshow("imgs",imgs)
            # cv2.namedWindow("imgt", cv2.NORM_MINMAX)
            # cv2.imshow("imgt", imgt)
            # cv2.namedWindow("seg", cv2.NORM_MINMAX)
            # cv2.imshow("seg", seg)
            #
            # key = cv2.waitKey()
            # if key==27:
            #     exit()