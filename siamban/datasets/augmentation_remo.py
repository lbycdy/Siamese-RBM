# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from siamban.utils.bbox import corner2center, \
        Center, center2corner, Corner
import random

def coveA(box1,box2):
    x1_1 = box1[0]
    y1_1 = box1[1]
    x2_1 = box1[2]
    y2_1 = box1[3]
    x1_2 = box2[0]
    y1_2 = box2[1]
    x2_2 = box2[2]
    y2_2 = box2[3]
    area_inter = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) *max((0, min(y2_1, y2_2) - max(y1_1, y1_2)))
    area1 = (x2_1 - x1_1)*(y2_1 - y1_1)
    return float(area_inter)/float(area1 + 1e-8)

class Augmentation:
    def __init__(self, shift, scale, blur, flip, color):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.flip = flip
        self.color = color
        self.rgbVar = np.array(
            [[-0.55919361,  0.98062831, - 0.41940627],
             [1.72091413,  0.19879334, - 1.82968581],
             [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):
        bbox = [float(x) for x in bbox]
        a = (out_sz-1) / (bbox[2]-bbox[0])
        b = (out_sz-1) / (bbox[3]-bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        image = cv2.filter2D(image, -1, kernel)
        return image

    def _color_aug(self, image):
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        return image

    def _gray_aug(self, image):
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return image

    def _shift_scale_aug(self, image, bbox, crop_bbox, size):
        im_h, im_w = image.shape[:2]
        bbox_center = corner2center(bbox)
        # adjust crop bounding box
        crop_bbox_center = corner2center(crop_bbox)
        # if self.scale:
        scale_x = (1.0 + Augmentation.random() * self.scale)
        scale_y = (1.0 + Augmentation.random() * self.scale)
        # scale_x = scale_y = 1.0-self.scale
        h, w = crop_bbox_center.h, crop_bbox_center.w
        scale_x = min(scale_x, float(im_w) / w)
        scale_y = min(scale_y, float(im_h) / h)
        crop_bbox_center = Center(crop_bbox_center.x,
                                  crop_bbox_center.y,
                                  crop_bbox_center.w * scale_x,
                                  crop_bbox_center.h * scale_y)

        cp_w,cp_h = crop_bbox_center.w,crop_bbox_center.h
        crop_bbox = center2corner(crop_bbox_center)
        cx_box,cy_box = bbox_center.x,bbox_center.y
        shiftleft = cx_box - cp_w
        shiftright = cx_box
        shifttop = cy_box - cp_h
        shiftbottom = cy_box
        shiftleft, shiftright,shifttop,shiftbottom = map(int,[shiftleft, shiftright,shifttop,shiftbottom])
        for i in range(20):
            if shiftleft<shiftright:
                x1 = random.randint(shiftleft,shiftright)
            else:
                x1 = crop_bbox.x1
            if shifttop<shiftbottom:
                y1 = random.randint(shifttop,shiftbottom)
            else:
                y1 = crop_bbox.y1
            x2 = x1 + cp_w
            y2 = y1 + cp_h
            if coveA(bbox,[x1,y1,x2,y2])>0.5:
                crop_bbox = Corner(x1, y1, x2, y2)
                break
        # adjust target bounding box
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                      bbox.x2 - x1, bbox.y2 - y1)

        if self.scale:
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_roi(image, crop_bbox, size)
        return image, bbox

    def _flip_aug(self, image, bbox):
        image = cv2.flip(image, 1)
        width = image.shape[1]
        bbox = Corner(width - 1 - bbox.x2, bbox.y1,
                      width - 1 - bbox.x1, bbox.y2)
        return image, bbox

    def __call__(self, image, bbox, size, gray=False):
        shape = image.shape
        crop_bbox = center2corner(Center(shape[0]//2, shape[1]//2,
                                         size-1, size-1))
        # gray augmentation
        if gray:
            image = self._gray_aug(image)

        # shift scale augmentation
        image, bbox = self._shift_scale_aug(image, bbox, crop_bbox, size)

        # color augmentation
        if self.color > np.random.random():
            image = self._color_aug(image)

        # blur augmentation
        if self.blur > np.random.random():
            image = self._blur_aug(image)

        # flip augmentation
        if self.flip and self.flip > np.random.random():
            image, bbox = self._flip_aug(image, bbox)
        return image, bbox

if __name__=='__main__':
    import json
    import os
    import copy


    def _get_bbox(image, shape):
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
    path_format = '{}.{}.{}.jpg'
    aug =  Augmentation(shift=64, scale=0.5, blur=0.0, flip = 0, color = 0)
    imgroot = "/home/ethan/GOT-10k_Train_000708"
    filejson = '/media/ethan/OldDisk/home/ethan/GOT/TrainData/GOT10K/train_withclassname.json'
    data=json.load(open(filejson))
    video = 'train/GOT-10k_Train_000708'
    track = '00'
    frame = "{:06d}".format(0)
    imgpath = os.path.join(imgroot,path_format.format(frame, track, 'x'))
    bbox = data[video][track][frame]
    img = cv2.imread(imgpath)
    bbox = _get_bbox(img,bbox)
    xmin,ymin,xmax,ymax = map(int,bbox)
    w = xmax - xmin
    h = ymax - ymin
    print("orgwh:",w,h)
    # cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),2)
    # cv2.namedWindow("img",cv2.NORM_MINMAX)
    # cv2.imshow("img",img)
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    w_list = []
    h_list = []
    for i in range(10000):
        b = copy.deepcopy(bbox)
        imgaug,box= aug(img.copy(),b,255)
        x1,y1,x2,y2 = map(int,box)
        cv2.rectangle(imgaug, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.namedWindow("imgaug", cv2.NORM_MINMAX)
        cv2.imshow("imgaug", imgaug)
        key = cv2.waitKey()
        if key==27:
            exit()
        xmin_list.append(x1)
        ymin_list.append(y1)
        xmax_list.append(x2)
        ymax_list.append(y2)
        w_list.append(x2-x1)
        h_list.append(y2-y1)
    print(min(xmin_list),max(xmax_list))
    print(min(ymin_list),max(ymax_list))
    print(min(w_list),max(w_list))
    print(min(h_list),max(h_list))


