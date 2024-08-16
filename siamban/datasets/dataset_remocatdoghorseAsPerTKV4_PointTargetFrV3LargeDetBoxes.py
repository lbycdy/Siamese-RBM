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
            return cls, delta,delta_weight

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

INF = 100000
class DetTarget:
    def __init__(self, ):
        print(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE_DET, cfg.TRAIN.SEARCH_DET_SIZE // 2)
        self.locations = self.compute_locations_per_level(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE_DET, cfg.TRAIN.SEARCH_DET_SIZE // 2)
        self.stride = cfg.POINT.STRIDE
        self.center_sampling_radius = cfg.DATASET.CATDOGHORSETK.DET_SAMPLERADIUS
        object_sizes_of_interest = [10,256]
        object_sizes_of_interest = self.locations.new_tensor(object_sizes_of_interest)
        self.object_sizes_of_interest = object_sizes_of_interest[None].expand(len(self.locations), -1)
    def compute_locations_per_level(self, stride,size,im_c):
        h,w = size,size
        ori = im_c - w // 2 * stride

        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + ori
        return locations

    def get_sample_region(self, gt, stride, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        stride = stride * radius
        xmin = center_x - stride
        ymin = center_y - stride
        xmax = center_x + stride
        ymax = center_y + stride
        # limit sample region in gt
        center_gt[:, :, 0] = torch.where(xmin > gt[:, :, 0], xmin, gt[:, :, 0])
        center_gt[:, :, 1] = torch.where(ymin > gt[:, :, 1], ymin, gt[:, :, 1])
        center_gt[:, :, 2] = torch.where(xmax > gt[:, :, 2],gt[:, :, 2], xmax)
        center_gt[:, :, 3] = torch.where(ymax > gt[:, :, 3],gt[:, :, 3], ymax)
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        xs, ys = locations[:, 0], locations[:, 1]
        if len(targets)>0:
            bboxes = targets[:, :4]
            labels_per_im = torch.tensor([1]*len(targets))
            area = targets[:, 4]

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)#torch.Size([2880, 1, 4])

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.stride,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0
        else:
            npos = len(xs)
            labels_per_im = torch.zeros((npos,))
            reg_targets_per_im = torch.zeros((npos,4))


        return labels_per_im, reg_targets_per_im

    def __call__(self, boxes):
        for i in range(len(boxes)):
            xmin,ymin,xmax,ymax = boxes[i]
            a = (xmax - xmin)*(ymax-ymin)
            boxes[i].append(a)
        target = torch.tensor(boxes).float()
        labels, reg_targets = self.compute_targets_for_locations(
            self.locations, target, self.object_sizes_of_interest
        )

        return labels, reg_targets


class BANDataset(Dataset):
    def __init__(self,world_size=None,batchsize=None):
        super(BANDataset, self).__init__()
        self.path_format = '{}.{}.{}.jpg'

        # create point target
        self.point_target = PointTarget()
        self.det_target = DetTarget()
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
        rootid = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            anno = subdata_cfg.ANNO
            root = subdata_cfg.ROOT

            # numuse_list = [subdata_cfg.NUM_USE_CAT,subdata_cfg.NUM_USE_DOG,subdata_cfg.NUM_USE_HORSE,subdata_cfg.NUM_USE_OTHER]
            # num_use_dog = subdata_cfg.NUM_USE_DOG
            # num_use_cat = subdata_cfg.NUM_USE_CAT
            # num_use_horse = subdata_cfg.NUM_USE_HORSE
            # num_use_other = subdata_cfg.NUM_USE_OTHER
            # flagcls_list = [False,False,False,False]
            data = json.load(open(anno))
            data_list = []
            data_list_video = []
            other_list = []
            cntins_dict = {}
            cntother = 0
            if name == "REMOCATDOGHORSE":
                data = json.load(open(anno))
                for imgpath in data:
                    if len(data[imgpath])>0:
                        boxes = []
                        for box in data[imgpath]:
                            xmin, ymin, xmax, ymax, cid = box
                            boxes.append([xmin,ymin,xmax,ymax])
                            try:
                                cntins_dict[cid] += 1
                            except:
                                cntins_dict[cid] = 1
                        data_list.append(['remo', rootid, imgpath,boxes])
            else:
                if name in ['COCO', 'DET']:
                    data = json.load(open(anno))
                    for imgpath in data:
                        imgpath_full  = os.path.join(root,imgpath)
                        if len(data[imgpath])>0 and os.path.exists(imgpath_full):
                            boxes = []
                            for box in data[imgpath]:
                                xmin, ymin, xmax, ymax, clstag = box
                                if str(clstag) in self.clstag2clsid_dict:
                                    cid = self.clstag2clsid_dict[str(clstag)]
                                    boxes.append([xmin,ymin,xmax,ymax])
                                    try:
                                        cntins_dict[cid] += 1
                                    except:
                                        cntins_dict[cid] = 1
                            if len(boxes)==0:#
                                cntother += len(data[imgpath])
                                other_list.append([name,rootid,imgpath,data[imgpath]])
                            else:
                                data_list.append([name, rootid, imgpath, boxes])
                else:
                    for video in data:
                        d_list = []
                        all_list = []
                        for track in data[video]:
                            imgpath_list = []
                            box_list = []
                            if name == 'LASOT':
                                c = os.path.dirname(video)
                            elif name == 'GOT10K':
                                c = data[video][track]['cls'][1].strip()
                            else:
                                c = data[video][track]['cls'].strip()
                            c = str(c)
                            frames = data[video][track]
                            # frames = list(map(int,filter(lambda x: x.isdigit(), frames.keys())))
                            frames = filter(lambda x: x.isdigit(), frames.keys())

                            for frame in frames:
                                box = data[video][track][frame]
                                imgpath = os.path.join(video, self.path_format.format(frame, track, 'x'))
                                imgpath_list.append(imgpath)
                                box_list.append(box)

                            boxinfo = [name, rootid, imgpath_list, box_list]
                            if c in self.clstag2clsid_dict:
                                cid =self.clstag2clsid_dict[c]
                                try:
                                    cntins_dict[cid] += 1
                                except:
                                    cntins_dict[cid] = 1
                                d_list.append(boxinfo)
                            else:
                                all_list.append(boxinfo)
                        if len(d_list)==0:
                            cntother += len(all_list)
                            other_list.extend(all_list)
                        else:
                            data_list_video.extend(d_list)
            numuse = max([subdata_cfg.NUM_USE_CAT,subdata_cfg.NUM_USE_DOG,subdata_cfg.NUM_USE_HORSE])
            numuse_other = subdata_cfg.NUM_USE_OTHER
            other_list = self.getnumdata(other_list, numuse_other)

            if name in ['REMOCATDOGHORSE','COCO','DET']:
                data_list = self.getnumdata(data_list, numuse)
                if len(data_list) > 0:
                    self.datalist_img.extend(data_list)
            else:
                data_list_video = self.getnumdata(data_list_video, numuse)
                if len(data_list_video) > 0:
                    self.datalist_vid.extend(data_list_video)
            if len(other_list) > 0:
                self.datalist_other.extend(other_list)
            info1 = "{} {} cat; {} dog; {} horse; {} other.(instance num for img;track num for video)".format(name,cntins_dict[1],cntins_dict[2],cntins_dict[3],cntother)
            info2 = "{} {} datalist_img, {} datalist_vid (numuse={}); {} datalist_other (numuse={}).".format(name,len(self.datalist_img),len(self.datalist_vid),numuse,len(self.datalist_other),numuse_other)
            logger.info(info1)
            logger.info(info2)
            print(info1)
            print(info2)

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
        self.resizedwh_det = cfg.TRAIN.SEARCH_DET_SIZE

        self.lambda_shift_ = 5
        self.lambda_scale_ = 15
        self.lambda_min_scale_ = -0.4
        self.lambda_max_scale_ = 0.4
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
        dsettag, rootid, imgpath_info, box_list = random.choice(self.datalist_other)
        if dsettag in ['COCO', 'DET']:
            image_path = os.path.join(self.root_list[rootid], imgpath_info)

            image = cv2.imread(image_path)
            boxref = random.choice(box_list)
            xmin, ymin, xmax, ymax = map(int, boxref[:4])
        else:
            idframe = random.randint(0,len(imgpath_info)-1)
            image_path = os.path.join(self.root_list[rootid], imgpath_info[idframe])
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
        idximg = index%self.num_img
        idxvid = index%self.num_vid
        imgs_prev_all = []
        imgs_curr_all = []
        boxes_gt_all = []
        label_posnegpair = []
        ###step1:get negative pairs of other object
        for i in range(cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHEROBJECT):
            imageother, box_other = self.get_otherobj()
            I_currs, box_gts,_,_ = self.MakeTrainingExamples(1, imageother, box_other)
            imgs_curr_all.append(I_currs[0])
            boxes_gt_all.append(box_gts[0])
            label_posnegpair.append(0)
        if random.uniform(0,1.0)<cfg.DATASET.CATDOGHORSETK.PROB_IMG:
            ###step2: get positive pairs of imagedata
            dsettag, rootid, imgpath, bbox_lists = self.datalist_img[idximg]
            img_prev = cv2.imread(os.path.join(self.root_list[rootid], imgpath))
            bbox_prev = random.choice(bbox_lists)
            img_curr = img_prev.copy()
            bbox_curr = bbox_prev[:]
            I_prev, imgs_curr, bboxes_gt,_,_ = self.MakeExamples(cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_IMAGE,
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
            bbox_prev = self._get_bbox(img_prev, bbox_prev)
            bbox_curr = self._get_bbox(img_curr, bbox_curr)
            I_prev, imgs_curr, bboxes_gt,_,_ = self.MakeExamples(cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_FRAME,
                                                                img_prev, img_curr, bbox_prev, bbox_curr)
            imgs_curr_all.extend(imgs_curr)
            boxes_gt_all.extend(bboxes_gt)
            label_posnegpair.extend([1]*len(imgs_curr))
        if random.uniform(0,1.0)<cfg.DATASET.CATDOGHORSETK.PROB_CROPTEMPLATE:
            sizehalf = self.resizedw_temp/2
            sx = random.uniform(cfg.DATASET.CATDOGHORSETK.SCALE_CROPTEMPLATE,1.0)
            sy = random.uniform(cfg.DATASET.CATDOGHORSETK.SCALE_CROPTEMPLATE,1.0)
            w_target = int(sizehalf*sx)
            h_target = int(sizehalf*sy)
            if w_target<sizehalf:
                xmin = 40+random.randint(0,sizehalf - w_target-1)
            else:
                xmin = 40
            if h_target < sizehalf:
                ymin = 40 + random.randint(0, sizehalf - h_target - 1)
            else:
                ymin = 40
            xmax = xmin + w_target
            ymax = ymin + h_target
            w4 = w_target//4
            h4 = h_target//4
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
            bbox = Corner(x1, y1, x2, y2)
            searchboxes.append([x1,y1,x2,y2])
            cls, delta = self.point_target(bbox, cfg.TRAIN.OUTPUT_SIZE, flagneg)
            cls_list.append(cls)
            delta_list.append(delta)

        I_prev = I_prev.transpose((2,0,1))
        search = np.stack(imgs_curr_all).transpose((0, 3, 1, 2))
        cls = np.stack(cls_list)
        delta = np.stack(delta_list)
        searchboxes = np.array(searchboxes)

        """"
        get detection data
        """
        search_det_list = []
        bboxes_det_list = []
        clsdet_list = []
        regdet_list = []
        for i in range(cfg.DATASET.CATDOGHORSETK.NUM_DET): #1
            dsettag, rootid, imgpath, bbox_lists = random.choice(self.datalist_img)
            img = cv2.imread(os.path.join(self.root_list[rootid], imgpath))
            image,boxes = self.getCropImage_Det(img,bbox_lists)
            image = Distortion(image)
            if random.uniform(0,1.0)<0.5:
                image = cv2.flip(image, 1)
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[j]
                    x1_new = self.resizedwh_det - x2
                    x2_new = self.resizedwh_det - x1
                    boxes[j] = [x1_new, y1, x2_new, y2]
            bboxes_det_list.append(boxes)
            search_det_list.append(image)
            clsdet,regdet = self.det_target(boxes)
            clsdet_list.append(clsdet)
            regdet_list.append(regdet)
        search_det = np.stack(search_det_list).transpose((0, 3, 1, 2))
        cls_det = torch.stack(clsdet_list, dim=0)
        reg_det = torch.stack(regdet_list, dim=0)


        return {
            'template': I_prev,
            'search': search,
            'label_cls': cls,
            'label_loc': delta,
            'searchboxes':searchboxes,
            'label_posnegpair': label_posnegpair,
            'search_det': search_det,
            # 'bboxes_det': bboxes_det_list,
            'cls_det': cls_det,
            'reg_det': reg_det,
        }
    def MakeExamples(self, num_generated_examples, image_prev, image_curr, bbox_prev, bbox_curr,seggt=None,boxes_all = None):
        imgs_curr = []
        bboxes_gt = []
        seggt_list = []
        bboxes_left = []
        I_prev = self.getprevimg(image_prev, bbox_prev)
        I_curr, box_gt,seg,boxes_true = self.MakeTrueExample(image_curr, bbox_prev, bbox_curr,seggt,boxes_all)
        imgs_curr.append(I_curr)
        bboxes_gt.append(box_gt)
        seggt_list.append(seg)
        bboxes_left.append(boxes_true)

        I_currs, box_gts,segs,boxes_tr = self.MakeTrainingExamples(num_generated_examples - 1, image_curr, bbox_curr,seggt,boxes_all)
        imgs_curr.extend(I_currs)
        bboxes_gt.extend(box_gts)
        seggt_list.extend(segs)
        bboxes_left.extend(boxes_tr)
        return I_prev, imgs_curr, bboxes_gt, seggt_list,bboxes_left

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

    def MakeTrainingExamples(self, num_generated_examples, image_curr, bbox_curr,seggt=None,boxes_all=None):
        imgs_curr = []
        bboxes_gt = []
        seg_crop = []
        boxes_left = []
        for i in range(num_generated_examples):
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
            else:
                seg_crop.append(None)


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
            if boxes_all is not None:
                boxes = []
                for box in boxes_all:
                    if self.coveA(box, roi_box) > 0.5:
                        xmin, ymin, xmax, ymax = box
                        xmin -= roi_x1
                        ymin -= roi_y1
                        xmax -= roi_x1
                        ymax -= roi_y1
                        xmin = xmin / (roi_x2 - roi_x1 + 1e-8) * self.resizedw_search
                        ymin = ymin / (roi_y2 - roi_y1 + 1e-8) * self.resizedw_search
                        xmax = xmax / (roi_x2 - roi_x1 + 1e-8) * self.resizedw_search
                        ymax = ymax / (roi_y2 - roi_y1 + 1e-8) * self.resizedw_search
                        boxes.append([xmin, ymin, xmax, ymax])
                boxes_left.append(boxes)
            else:
                boxes_left.append(None)

        return imgs_curr, bboxes_gt, seg_crop,boxes_left

    def MakeTrueExample(self, image_curr, bbox_prev, bbox_curr,seggt=None,boxes_all=None):

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
        else:
            segcrop = None
        if boxes_all is not None:
            boxes_left = []
            for box in boxes_all:
                if self.coveA(box,roi_box)>0.5:
                    xmin,ymin,xmax,ymax = box
                    xmin -= roi_x1
                    ymin -= roi_y1
                    xmax -= roi_x1
                    ymax -= roi_y1
                    xmin = xmin /(roi_x2 - roi_x1 + 1e-8)*self.resizedw_search
                    ymin = ymin /(roi_y2 - roi_y1 + 1e-8)*self.resizedw_search
                    xmax = xmax /(roi_x2 - roi_x1 + 1e-8)*self.resizedw_search
                    ymax = ymax /(roi_y2 - roi_y1 + 1e-8)*self.resizedw_search
                    boxes_left.append([xmin,ymin,xmax,ymax])
        else:
            boxes_left = None
        return img_curr, [gt_x1, gt_y1, gt_x2, gt_y2], segcrop,boxes_left

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
    def checkcropbox_ok(self,cropbox,imagew,imageh):
        cpxmin, cpymin, cpxmax, cpymax = map(int,cropbox)
        # padleft = 0 if cpxmin > 0 else -cpxmin
        # padtop = 0 if cpymin > 0 else -cpymin
        # padright = 0 if cpxmax < imagew else cpxmax - imagew
        # padbottom = 0 if cpymax < imageh else cpymax - imageh
        xmin = max(0, cpxmin)
        ymin = max(0, cpymin)
        xmax = min(cpxmax, imagew)
        ymax = min(cpymax, imageh)
        w = xmax - xmin
        h = ymax - ymin
        flag = w>0 and h>0
        return flag



    def getCropImage_Det(self, image, boxes):
        imageh, imagew = image.shape[:2]
        crop_pos = []
        nb = len(boxes)
        iou_limits = [0.5, 0.7, 0.9]
        for _ in range(5):
            boxref = random.choice(boxes)
            xminref, yminref, xmaxref, ymaxref = boxref
            cp_w = (xmaxref - xminref) * 2*self.resizedwh_det/float(self.resizedw_temp)
            cp_h = (ymaxref - yminref) * 2*self.resizedwh_det/float(self.resizedw_temp)
            for _ in range(3):
                sx = self.lambda_min_scale_ + random.uniform(0, 1.0) * (self.lambda_max_scale_ - self.lambda_min_scale_)
                sy = self.lambda_min_scale_ + random.uniform(0, 1.0) * (self.lambda_max_scale_ - self.lambda_min_scale_)
                cropw = cp_w*sx
                croph = cp_h*sy

                for i in range(len(iou_limits)):
                    iou_limit = iou_limits[i]
                    flagok = False
                    for _ in range(20):
                        cropw, croph = map(int, [cropw, croph])
                        cropxmin = random.uniform(0, 1.0) * (imagew - cropw)
                        cropymin = random.uniform(0, 1.0) * (imageh - croph)
                        cropxmax = cropxmin + cropw
                        cropymax = cropymin + croph
                        for ib in range(nb):
                            xmin, ymin, xmax, ymax = boxes[ib][:4]
                            cova = self.coveA([xmin, ymin, xmax, ymax], [cropxmin, cropymin, cropxmax, cropymax])
                            if cova > iou_limit and self.checkcropbox_ok([cropxmin, cropymin, cropxmax, cropymax],imagew,imageh):
                                flagok = True
                                break
                        if flagok:
                            break
                    if flagok:
                        crop_pos.append([cropxmin, cropymin, cropw, croph])
        if len(crop_pos) > 0:
            cropxmin, cropymin, cropw, croph = random.choice(crop_pos)
        else:
            boxref = random.choice(boxes)
            xminref, yminref, xmaxref, ymaxref = boxref
            cropw = (xmaxref - xminref) * 2 * self.resizedwh_det / float(self.resizedw_temp)
            croph = (ymaxref - yminref) * 2 * self.resizedwh_det / float(self.resizedw_temp)
            cx = (xminref+xmaxref)/2
            cy = (yminref+ymaxref)/2
            cropxmin = cx - cp_w/2
            cropymin = cy - cp_h/2

        cropxmax = cropxmin + cropw
        cropymax = cropymin + croph
        cropxmin, cropymin, cropxmax, cropymax = map(int, [cropxmin, cropymin, cropxmax, cropymax])
        padleft = 0 if cropxmin > 0 else -cropxmin
        padtop = 0 if cropymin > 0 else -cropymin
        padright = 0 if cropxmax < imagew else cropxmax - imagew
        padbottom = 0 if cropymax < imageh else cropymax - imageh
        xmin = max(0, cropxmin)
        ymin = max(0, cropymin)
        xmax = min(cropxmax, imagew)
        ymax = min(cropymax, imageh)
        # xmin,ymin,xmax,ymax = map(int,[xmin,ymin,xmax,ymax])
        imgcrop = image[ymin:ymax, xmin:xmax, :]
        img_pad = cv2.copyMakeBorder(imgcrop, padtop, padbottom, padleft, padright, cv2.BORDER_CONSTANT,
                                     value=(117, 117, 117))
        if img_pad is None:
            img_pad = image
            cropxmin = cropymin = 0
            cropxmax,cropymax = imagew,imageh
            print(imgcrop.shape,padtop,padbottom,padleft,padright)
            print(cropxmin,cropymin,cropxmax,cropymax,xmin,ymin,xmax,ymax,image.shape)
        imghpad, imgwpad = img_pad.shape[:2]
        img_pad = cv2.resize(img_pad, (self.resizedwh_det, self.resizedwh_det))
        sx = float(self.resizedwh_det) / imgwpad
        sy = float(self.resizedwh_det) / imghpad

        boxes_left = []
        cova_list = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            cova = self.coveA([xmin, ymin, xmax, ymax], [cropxmin, cropymin, cropxmax, cropymax])
            cova_list.append(cova)
            if cova > 0.25:
                x1 = (xmin - cropxmin) * sx
                x2 = (xmax - cropxmin) * sx
                y1 = (ymin - cropymin) * sy
                y2 = (ymax - cropymin) * sy
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, self.resizedwh_det)
                y2 = min(y2, self.resizedwh_det)
                boxes_left.append([x1, y1, x2, y2])
        if len(boxes_left) == 0:
            idx = cova_list.index(max(cova_list))
            xmin, ymin, xmax, ymax = boxes[idx]
            x1 = (xmin - cropxmin) * sx
            x2 = (xmax - cropxmin) * sx
            y1 = (ymin - cropymin) * sy
            y2 = (ymax - cropymin) * sy
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, self.resizedwh_det)
            y2 = min(y2, self.resizedwh_det)
            boxes_left.append([x1, y1, x2, y2])

        return img_pad, boxes_left

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

    import socket
    #
    hostname = socket.gethostname()
    """"
    get qt data
    """
    cfgfile = "/home/inspur/work/siamban_GOTREMO/experiments/siamban_r50_l234/20221213DogCatHorseAsPersonV004MobileOneWeightAdd_ACMOutPointMask_FrV003LargeDet_4GPU128.yaml"
    cfg.merge_from_file(cfgfile)
    cfg.DATASET.CATDOGHORSETK.FLIP_SEARCH = 0.2
    cfg.DATASET.CATDOGHORSETK.FLIPVER_SEARCH = 0.2
    # cfg.TRAIN.SEARCH_DET_SIZE = 224
    # cfg.TRAIN.OUTPUT_SIZE_DET = 19
    # cfg.DATASET.CATDOGHORSETK.NUM_DET = 5
    print(cfg)
    numneg = cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHEROBJECT
    numpos = cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_IMAGE
    data = BANDataset()
    idlist = list(range(len(data)))
    random.shuffle(idlist)
    cnt = 0
    for i in idlist[:100]:
        a = data.__getitem__(i)
        search = a["search"]
        template = a['template'].transpose((1,2,0))
        searchboxes = a['searchboxes']
        bboxes_det = a['bboxes_det']
        # if len(bboxes_det)>0:
        #     # idxneg = random.randint(0,numneg-1)
        #     # sneg=search[idxneg].transpose((1,2,0))
        #     # idxpos = numneg+random.randint(0,numpos-1)
        #     # spos = search[idxpos].transpose((1, 2, 0))
        #     # xmin,ymin,xmax,ymax = map(int,searchboxes[idxneg])
        #     # cv2.rectangle(sneg,(xmin,ymin),(xmax,ymax),(0,0,255),1)
        #     # cv2.rectangle(template,(40,40),(120,120),(0,0,255),2)
        #     # cv2.imwrite("images/%06dneg.z.jpg"%cnt,template)
        #     # cv2.imwrite("images/%06dneg.x.jpg"%cnt,sneg)
        #     # cnt += 1
        #     # xmin, ymin, xmax, ymax = map(int, searchboxes[idxpos])
        #     # cv2.rectangle(spos, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        #     # cv2.rectangle(template,(40,40),(120,120),(0,0,255),2)
        #     #
        #     # cv2.imwrite("images/%06dpos.z.jpg" % cnt, template)
        #     # cv2.imwrite("images/%06dpos.x.jpg" % cnt, spos)
        #     # cnt += 1
        #
        #     for j in range(len(bboxes_det)):
        #         spos = search[j+numneg].transpose((1, 2, 0))
        #         imgh,imgw = spos.shape[:2]
        #         for box in bboxes_det[j]:
        #             xmin,ymin,xmax,ymax = box
        #             xmin,ymin,xmax,ymax = map(int,[xmin,ymin,xmax,ymax])
        #             xmin = max(0,xmin)
        #             ymin = max(0,ymin)
        #             xmax = min(xmax,imgw)
        #             ymax = min(ymax,imgh)
        #             cv2.rectangle(spos, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        #         cv2.imwrite("images/%06dpos.x.jpg" % cnt, spos)
        #         cnt += 1
        search_det = a['search_det']
        bboxes_det = a['bboxes_det']
        reg_det = a['reg_det'].data.numpy()
        cls_det = a['cls_det'].data.numpy()
        print(cls_det.shape,reg_det.shape,search_det.shape)
        exit()
        for i in range(len(search_det)):
            spos = search_det[i].transpose((1, 2, 0))
            cls_showorg = cls_det[i].reshape((19,19))
            flag = cls_showorg>0
            print(cls_showorg.min(),cls_showorg.max(),flag.sum())
            cls_showorg *= 255
            cls_showorg = cls_showorg.astype(np.uint8)
            head1 = cv2.resize(cls_showorg, (cfg.TRAIN.SEARCH_DET_SIZE, cfg.TRAIN.SEARCH_DET_SIZE))
            head1 = head1.astype(np.uint8)
            heat_img = cv2.applyColorMap(head1, cv2.COLORMAP_JET)
            spos = cv2.addWeighted(spos, 0.5, heat_img, 0.5, 0)
            for box in bboxes_det[i]:
                print(box)
                xmin, ymin, xmax, ymax = map(int, box[:4])
                cv2.rectangle(spos, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            cv2.imwrite("images/%06ddet.x.jpg" % cnt, spos)
            cnt += 1



    exit()
    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # s.connect(("8.8.8.8", 80))
    # ipstr = s.getsockname()[0]
    # ipaddress_int = int(ipstr.split('.')[-1])
    # if ipaddress_int==111:
    #     cfg.DATASET.CATDOGHORSETK.DATASET_ROOT = '/home/zhangming/Datasets/GOT/PersonTracker'
    #     cfg.DATASET.CATDOGHORSETK.IMAGE_LIST = ('/home/zhangming/Datasets/GOT/PersonTracker/Layout/persontracker_imagedata_cocovoc_20220802.json',)
    #     cfg.DATASET.CATDOGHORSETK.VID_LIST = ('/home/zhangming/Datasets/GOT/PersonTracker/Layout/alov.json',
    #                                      '/home/zhangming/Datasets/GOT/PersonTracker/Layout/otb.json',
    #                                      '/home/zhangming/Datasets/GOT/PersonTracker/Layout/TrackerNet.json',
    #                                      '/home/zhangming/Datasets/GOT/PersonTracker/Layout/VOT2016.json',
    #                                      '/home/zhangming/Datasets/GOT/PersonTracker/Layout/Youtube.json',
    #                                      )
    # elif ipaddress_int==110:
    #     cfg.DATASET.CATDOGHORSETK.DATASET_ROOT = '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/PersonTracker'
    #     cfg.DATASET.CATDOGHORSETK.IMAGE_LIST = (
    #     '/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/PersonTracker/Layout/persontracker_imagedata_cocovoc_20220802.json',)
    #     cfg.DATASET.CATDOGHORSETK.VID_LIST = ('/home/zhangming/de33339e-24e6-48b1-883a-5eb29fb8c18e/Datasets/GOT/PersonTracker/Layout/alov.json',
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
    # cfg.DATASET.CATDOGHORSETK.KCONTEXTFACTOR = 2.0
    # cfg.DATASET.CATDOGHORSETK.TYPE_CONTEXTBOX = 0
    # cfg.DATASET.CATDOGHORSETK.VIDEOS_PER_EPOCH = 600000
    # cfg.DATASET.CATDOGHORSETK.CHANGEPADDING_VALUE = (104, 117, 123)
    # cfg.DATASET.CATDOGHORSETK.GRAY = 0.0
    # cfg.DATASET.CATDOGHORSETK.FETCH_ITERS = 2
    # cfg.TRAIN.SEARCH_SIZE = 160
    # cfg.TRAIN.OUTPUT_SIZE = 11
    # cfg.DATASET.FLAG_CHECKDESIRED_OUTPUT = False
    # cfg.ANCHOR.STRIDE = 8
    # cfg.ANCHOR.RATIOS = [0.5,1,2.0]
    # cfg.ANCHOR.SCALES = [11]
    # cfg.ANCHOR.ANCHOR_NUM = 3
    #
    # cfg.DATASET.CATDOGHORSETK.FLIP_TEMPLATE = 0.0
    # cfg.DATASET.CATDOGHORSETK.FLIPVER_TEMPLATE = 0.0
    # cfg.DATASET.CATDOGHORSETK.FLIP_SEARCH = 0.0
    # cfg.DATASET.CATDOGHORSETK.FLIPVER_SEARCH = 0.0

    cfgfile = "/home/inspur/work/siamban-acm/experiments/siamban_r50_l234/20221128DogCatHorseAsPersonV003MobileOneS8S16S32Add_ACMOutPointMaskDV3VIRandPNMod1SFTMod1SIn224_4GPU127.yaml"
    cfg.merge_from_file(cfgfile)
    cfg.DATASET.CATDOGHORSETK.PROB_CROPTEMPLATE = 1.0
    cfg.TRAIN.SEARCH_DET_SIZE = 256
    cfg.DATASET.CATDOGHORSETK.NUM_DET = 3
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

        #
        # step = cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHEROBJECT+cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_IMAGE + cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_FRAME
        # rangelist = [0,cfg.DATASET.CATDOGHORSETK.NUM_NEG_OTHEROBJECT,cfg.DATASET.CATDOGHORSETK.NUM_GENERATE_PER_IMAGE]
        # rangelist = list(range(step))
        # nbatch = nimg//step
        # for ib in range(nbatch):
        #     base = ib*step
        #     for k in rangelist:
        #         j = base + k
        #         imgs = search[j].transpose((1, 2, 0))
        #         imgt = template[j].transpose((1, 2, 0))
        #         loc = boxes_gt_all[j]
        #         x1,y1,x2,y2 = loc
        #         x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
        #         if label_posnegpair[j]==1:
        #             cv2.rectangle(imgs,(x1,y1),(x2,y2),(0,0,255))
        #         else:
        #             cv2.rectangle(imgs,(x1,y1),(x2,y2),(0,255,0))
                # if label_posnegpair[j] == 1:
                #     cv2.imwrite("images/%06dpos.z.jpg"%cnt,imgt)
                #     cv2.imwrite("images/%06dpos.x.jpg"%cnt,imgs)
                # else:
                #     cv2.imwrite("images/%06dneg.z.jpg" % cnt, imgt)
                #     cv2.imwrite("images/%06dneg.x.jpg" % cnt, imgs)
                # cnt += 1

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




    def shift_bbox(image, box):  # box = [xmin,ymin,xmax,ymax]
        shift_motion_model = True
        lambda_shift_ = 5
        lambda_scale_ = 15
        lambda_min_scale_ = -0.4
        lambda_max_scale_ = 0.4
        kContextFactorShiftBox = 3.0
        img_h, img_w = image.shape[:2]
        width = box[2] - box[0]
        height = box[3] - box[1]
        center_x = (box[2] + box[0]) / 2.0
        center_y = (box[3] + box[1]) / 2.0
        kMaxNumTries = 10
        num_tries_width = 0
        new_width = -1
        while ((new_width < 0 or new_width > img_w - 1) and num_tries_width < kMaxNumTries):
            if shift_motion_model:
                smp_d = sample_exp_two_sided(lambda_scale_)
                width_scale_factor = max(lambda_min_scale_, min(lambda_max_scale_, smp_d))
            else:
                rand_num = random.random()
                width_scale_factor = lambda_min_scale_ + rand_num(
                    lambda_max_scale_ - lambda_min_scale_)
            new_width = width * (1 + width_scale_factor)
            new_width = max(1.0, min(img_w - 1, new_width))
            num_tries_width += 1

        num_tries_height = 0
        new_height = -1
        while ((new_height < 0 or new_height > img_h - 1) and num_tries_height < kMaxNumTries):
            if shift_motion_model:
                smp_d = sample_exp_two_sided(lambda_scale_)
                height_scale_factor = max(lambda_min_scale_, min(lambda_max_scale_, smp_d))
            else:
                rand_num = random.random()
                height_scale_factor = lambda_min_scale_ + rand_num(
                    lambda_max_scale_ - lambda_min_scale_)
            new_height = height * (1 + height_scale_factor)
            new_height = max(1.0, min(img_h - 1, new_height))
            num_tries_height += 1

        first_time_x = True
        new_center_x = -1
        num_tries_x = 0
        while ((first_time_x or
                new_center_x < center_x - width * kContextFactorShiftBox / 2 or
                new_center_x > center_x + width * kContextFactorShiftBox / 2 or
                new_center_x - new_width / 2 < 0 or
                new_center_x + new_width / 2 > img_w) and
               num_tries_x < kMaxNumTries):
            if shift_motion_model:
                smp_d = sample_exp_two_sided(lambda_shift_)
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
                new_center_y < center_y - height * kContextFactorShiftBox / 2 or
                new_center_y > center_y + height * kContextFactorShiftBox / 2 or
                new_center_y - new_height / 2 < 0 or
                new_center_y + new_height / 2 > img_h) and
               num_tries_y < kMaxNumTries):
            if shift_motion_model:
                smp_d = sample_exp_two_sided(lambda_shift_)
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

    def sample_exp_two_sided(lam):
        prob = random.random()
        if prob > 0.5:
            pos_or_neg = 1
        else:
            pos_or_neg = -1
        rand_uniform = random.random()
        return math.log(rand_uniform) / lam * pos_or_neg

    img = np.random.random((224,224,3))
    box = [40,40,120,120]
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    for i in range(1000):
        xmin,ymin,xmax,ymax = shift_bbox(img,box)
        xmin_list.append(xmin)
        ymin_list.append(ymin)
        xmax_list.append(xmax)
        ymax_list.append(ymax)
    print(min(xmin_list),min(xmin_list),max(xmax_list),max(ymax_list))

