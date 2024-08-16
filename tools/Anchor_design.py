# -*- coding:utf-8 -*-
"""
@Time: 2023/2/22 上午11:31
@Author: wudengyang
@Description:
    1. show gt w, h, area...
    2. show anchor w, h
    3. show anchor match num
"""
import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from siamban.utils.bbox import corner2center, center2corner
from collections import Counter
import tqdm


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

def shift_bbox_remo2(box):  # box = [xmin,ymin,xmax,ymax]
    """版本

    self.lambda_min_scale_ = -0.4
    self.lambda_max_scale_ = 0.4
    """
    lambda_max_scale_ = 0.5
    lambda_min_scale_ = -0.5
    lambda_max_ratio_ = 0.2
    lambda_min_ratio_ = -0.1
    xmin, ymin, xmax, ymax = box
    cp_w = (xmax - xmin) * 2
    cp_h = (ymax - ymin) * 2
    cx_box = (xmin + xmax) / 2
    cy_box = (ymax + ymin) / 2

    s_rand = lambda_min_scale_ + random.uniform(0, 1.0) * (lambda_max_scale_ - lambda_min_scale_)
    r_rand = lambda_min_ratio_ + random.uniform(0, 1.0) * (lambda_max_ratio_ - lambda_min_ratio_)
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
        if coveA(box, [x1, y1, x2, y2]) > 0.3:
            flagok = True
            break
    if not flagok:
        # cp_w = (xmax - xmin) * 2
        # cp_h = (ymax - ymin) * 2
        # cx_box = (xmin + xmax) / 2
        # cy_box = (ymax + ymin) / 2
        x1 = cx_box - cp_w / 2
        y1 = cy_box - cp_h / 2
        x2 = x1 + cp_w
        y2 = y1 + cp_h
    return x1, y1, x2, y2, flagok
def shift_bbox_remo7(box):  # box = [xmin,ymin,xmax,ymax]
    """

    self.lambda_min_scale_ = -0.4
    self.lambda_max_scale_ = 0.4
    """
    lambda_max_scale_ = 0.5
    lambda_min_scale_ = -0.5
    lambda_max_ratio_ = 0.3
    lambda_min_ratio_ = -0.3
    lambda_scale_ = 10.0
    xmin, ymin, xmax, ymax = box
    bw = xmax - xmin
    bh = ymax - ymin
    cp_w = bw * 2
    cp_h = bh * 2

    cx_box = (xmin + xmax) / 2
    cy_box = (ymax + ymin) / 2

    s_rand = lambda_min_scale_ + random.uniform(0, 1.0) * (lambda_max_scale_ - lambda_min_scale_)
    r_rand = lambda_min_ratio_ + random.uniform(0, 1.0) * (lambda_max_ratio_ - lambda_min_ratio_)
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
        if coveA(box, [x1, y1, x2, y2]) > 0.35:
            flagok = True
            break
    if not flagok:
        x1 = cx_box - cp_w / 2
        y1 = cy_box - cp_h / 2
        x2 = x1 + cp_w
        y2 = y1 + cp_h
    return [x1, y1, x2, y2]
def shift_bbox_remo4(box):  # box = [xmin,ymin,xmax,ymax]
    """

    self.lambda_min_scale_ = -0.4
    self.lambda_max_scale_ = 0.4
    """
    lambda_max_scale_ = 0.5
    lambda_min_scale_ = -0.5
    lambda_max_ratio_ = 0.2
    lambda_min_ratio_ = -0.2
    lambda_scale_ = 10.0
    xmin, ymin, xmax, ymax = box
    cp_w = (xmax - xmin) * 2
    cp_h = (ymax - ymin) * 2
    cx_box = (xmin + xmax) / 2
    cy_box = (ymax + ymin) / 2

    r_rand = lambda_min_ratio_ + random.uniform(0, 1.0) * (lambda_max_ratio_ - lambda_min_ratio_)
    s_rand = math.log(random.random()) / lambda_scale_
    s_rand = max(lambda_min_scale_, min(lambda_max_scale_, s_rand))

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
        if coveA(box, [x1, y1, x2, y2]) > 0.65:
            flagok = True
            break
    if not flagok:
        x1 = cx_box - cp_w / 2
        y1 = cy_box - cp_h / 2
        x2 = x1 + cp_w
        y2 = y1 + cp_h
    return x1, y1, x2, y2, flagok

def imshow_w_h_area(N):
    box = [80, 80, 160, 160]
    count_remo2 = np.zeros((160, 160))
    count_remo4 = np.zeros((160, 160))
    whlist_remo2 = []
    wlist_remo2 = []
    hlist_remo2 = []
    cov_list_remo2 = []
    covin_img_remo2 = []
    whlist_remo4 = []
    wlist_remo4 = []
    hlist_remo4 = []
    cov_list_remo4 = []
    covin_img_remo4 = []
    for i in range(N):
        x1c, y1c, x2c, y2c, flagok = shift_bbox_remo2(box)
        wcrop = x2c - x1c
        hcrop = y2c - y1c
        sx = 160.0 / wcrop
        sy = 160.0 / hcrop
        xmin, ymin, xmax, ymax = box
        xmin -= x1c
        xmax -= x1c
        ymin -= y1c
        ymax -= y1c
        xmin *= sx
        ymin *= sy
        xmax *= sx
        ymax *= sy
        bw = xmax - xmin
        bh = ymax - ymin
        whlist_remo2.append(bw / bh)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(160, xmax)
        ymax = min(160, ymax)
        w = xmax - xmin
        h = ymax - ymin
        wlist_remo2.append(w)
        hlist_remo2.append(h)
        xc = int((xmin + xmax) / 2)
        yc = int((ymin + ymax) / 2)
        if xc > 0 and yc > 0 and xc < 160 and yc < 160:
            count_remo2[yc, xc] += 1
        c = coveA(box, [x1c, y1c, x2c, y2c])
        cov_list_remo2.append(c)
        c = coveA([x1c, y1c, x2c, y2c], box)
        covin_img_remo2.append(c)

        # x1c, y1c, x2c, y2c, flagok = shift_bbox_remo4(box)
        # wcrop = x2c - x1c
        # hcrop = y2c - y1c
        # sx = 160.0 / wcrop
        # sy = 160.0 / hcrop
        # xmin, ymin, xmax, ymax = box
        # xmin -= x1c
        # xmax -= x1c
        # ymin -= y1c
        # ymax -= y1c
        # xmin *= sx
        # ymin *= sy
        # xmax *= sx
        # ymax *= sy
        # bw = xmax - xmin
        # bh = ymax - ymin
        # whlist_remo4.append(bw / bh)
        # xmin = max(0, xmin)
        # ymin = max(0, ymin)
        # xmax = min(160, xmax)
        # ymax = min(160, ymax)
        # w = xmax - xmin
        # h = ymax - ymin
        # wlist_remo4.append(w)
        # hlist_remo4.append(h)
        # xc = int((xmin + xmax) / 2)
        # yc = int((ymin + ymax) / 2)
        # if xc > 0 and yc > 0 and xc < 160 and yc < 160:
        #     count_remo4[yc, xc] += 1
        # c = coveA(box, [x1c, y1c, x2c, y2c])
        # cov_list_remo4.append(c)
        # c = coveA([x1c, y1c, x2c, y2c], box)
        # covin_img_remo4.append(c)

    count_remo2 /= count_remo2.max()
    count_remo2 *= 255
    count_remo2 = count_remo2.astype(np.uint8)
    cv2.namedWindow("count_remo2", cv2.NORM_HAMMING)
    cv2.imshow("count_remo2", count_remo2)
    # count_remo4 /= count_remo4.max()
    # count_remo4 *= 255
    # count_remo4 = count_remo4.astype(np.uint8)
    # cv2.namedWindow("count_remo4", cv2.NORM_HAMMING)
    # cv2.imshow("count_remo4", count_remo4)
    key = cv2.waitKey()

    plt.figure()
    cov_list_remo2.sort()
    plt.plot(cov_list_remo2, label="cov_list_remo2")
    # cov_list_remo4.sort()
    # plt.plot(cov_list_remo4, label="cov_list_remo4")
    plt.legend()

    plt.figure()
    covin_img_remo2.sort()
    plt.plot(covin_img_remo2, label="covin_img_remo2")
    # covin_img_remo4.sort()
    # plt.plot(covin_img_remo4, label="covin_img_remo4")
    plt.legend()

    plt.figure()
    whlist_remo2.sort()
    plt.plot(whlist_remo2, label="whlist_remo2")
    # whlist_remo4.sort()
    # plt.plot(whlist_remo4, label="whlist_remo4")
    plt.legend()

    plt.figure()
    wlist_remo2.sort()
    plt.plot(wlist_remo2, label="wlist_remo2")
    # wlist_remo4.sort()
    # plt.plot(wlist_remo4, label="wlist_remo4")
    plt.legend()

    plt.figure()
    hlist_remo2.sort()
    plt.plot(hlist_remo2, label="hlist_remo2")
    # hlist_remo4.sort()
    # plt.plot(hlist_remo4, label="hlist_remo4")
    plt.legend()

    plt.show()

def imshow_anchors(RATIOS, SCALES, STRIDE = 8):
    img = np.zeros((160, 160, 3))
    colorlist = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
    size = STRIDE * STRIDE
    i = 0
    wh_list = []
    for r in RATIOS:
        ws = int(math.sqrt(size * 1. / r))
        hs = int(ws * r)
        for s in SCALES:
            w = ws * s
            h = hs * s
            wh_list.append([w, h])
            # print(f'r:{r}, s:{s}, w:{w}, h:{h}')
            anchor = [-w / 2 + 80, -h / 2 + 80, w / 2 + 80, h / 2 + 80]
            anchor = list(map(int, anchor))
            cv2.rectangle(img, (anchor[0], anchor[1]), (anchor[2], anchor[3]), colorlist[i], 1)
            i += 1
    # cv2.namedWindow('img', cv2.NORM_HAMMING)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # cv2.imwrite(f'{RATIOS},{SCALES}.jpg', img)
    for wh in wh_list:
        print(f'{wh}:{wh[0] * wh[1] / 160 / 160}')


def IoU(rect1, rect2):
    """ caculate interection over union
    Args:
        rect1: (x1, y1, x2, y2)
        rect2: (x1, y1, x2, y2)
    Returns:
        iou
    """
    # overlap
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2-x1) * (y2-y1)
    target_a = (tx2-tx1) * (ty2 - ty1)
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou

class Anchors:
    """
    This class generate anchors.
    """
    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = image_center
        self.size = size

        self.anchor_num = len(self.scales) * len(self.ratios)

        self.anchors = None

        self.generate_anchors()


    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size*1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w*0.5, -h*0.5, w*0.5, h*0.5][:]
                count += 1

    def generate_all_anchors(self, im_c, size):
        """
        im_c: image center
        size: image size
        """
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size
        #im_c = 255/2, size = 25,stride = 8
        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]
        # print(zero_anchors)
        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1),
                             [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                            np.stack([cx, cy, w,  h]).astype(np.float32))
        # print(self.all_anchors[0].shape,self.all_anchors[1].shape)
        # print(self.all_anchors[0][:,0,2,2])
        return True

class AnchorTarget:
    def __init__(self,RATIOS, SCALES, SEARCH_SIZE=160, STRIDE=8, OUTPUT_SIZE=19):
        self.anchors = Anchors(STRIDE,
                               RATIOS,
                               SCALES)

        self.anchors.generate_all_anchors(im_c=SEARCH_SIZE//2,size=OUTPUT_SIZE)


    def __call__(self, target, size, neg=False, SEARCH_SIZE=160, STRIDE=8, NEG_NUM=24, POS_NUM=24, TOTAL_NUM=96, THR_HIGH=0.6, THR_LOW=0.3):
        anchor_num = len(RATIOS) * len(SCALES)

        # -1 ignore 0 negative 1 positive
        cls = 0 * np.ones((anchor_num, size, size), dtype=np.int64)
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)

        if neg:
            # l = size // 2 - 3
            # r = size // 2 + 3 + 1
            # cls[:, l:r, l:r] = 0

            cx = size // 2
            cy = size // 2
            cx += int(np.ceil((tcx - SEARCH_SIZE // 2) /
                      STRIDE + 0.5))
            cy += int(np.ceil((tcy - SEARCH_SIZE // 2) /
                      STRIDE + 0.5))
            l = max(0, cx - 3)
            r = min(size, cx + 4)
            u = max(0, cy - 3)
            d = min(size, cy + 4)
            cls[:, u:d, l:r] = 0

            # neg, neg_num = select(np.where(cls == 0), NEG_NUM)
            # cls[:] = -1
            # cls[neg] = 0

            overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
            return cls, delta, delta_weight, overlap

        anchor_box = self.anchors.all_anchors[0]
        anchor_center = self.anchors.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]

        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > THR_HIGH)
        neg = np.where(overlap < THR_LOW)
        # flagposzero = len(pos[0])==0
        # if flagposzero:
        # print(len(pos[0]),len(neg[0]),overlap.max())
        # pos, pos_num = select(pos, POS_NUM)
        # pos_num = len(pos)
        # neg, neg_num = select(neg, TOTAL_NUM - POS_NUM)


        cls[pos] = 1

        # delta_weight[pos] = 1. / (pos_num + 1e-6)

        cls[neg] = 0
        return cls, delta, delta_weight, overlap, Counter(pos[0])

class AnchorTarget1:
    def __init__(self,RATIOS, SCALES, SEARCH_SIZE=160, STRIDE=8, OUTPUT_SIZE=19):
        self.anchors = Anchors(STRIDE,
                               RATIOS,
                               SCALES)

        self.anchors.generate_all_anchors(im_c=SEARCH_SIZE//2,size=OUTPUT_SIZE)


    def __call__(self, target, size, neg=False, SEARCH_SIZE=160, STRIDE=8, NEG_NUM=24, POS_NUM=24, TOTAL_NUM=96, THR_HIGH=0.6, THR_LOW=0.3):
        anchor_num = len(RATIOS) * len(SCALES)

        # -1 ignore 0 negative 1 positive
        cls = 0 * np.ones((anchor_num, size, size), dtype=np.int64)
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)

        if neg:
            # l = size // 2 - 3
            # r = size // 2 + 3 + 1
            # cls[:, l:r, l:r] = 0

            cx = size // 2
            cy = size // 2
            cx += int(np.ceil((tcx - SEARCH_SIZE // 2) /
                      STRIDE + 0.5))
            cy += int(np.ceil((tcy - SEARCH_SIZE // 2) /
                      STRIDE + 0.5))
            l = max(0, cx - 3)
            r = min(size, cx + 4)
            u = max(0, cy - 3)
            d = min(size, cy + 4)
            cls[:, u:d, l:r] = 0

            # neg, neg_num = select(np.where(cls == 0), NEG_NUM)
            # cls[:] = -1
            # cls[neg] = 0

            overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
            return cls, delta, delta_weight, overlap

        anchor_box = self.anchors.all_anchors[0]
        anchor_center = self.anchors.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]

        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > THR_HIGH)
        neg = np.where(overlap < THR_LOW)
        # flagposzero = len(pos[0])==0
        # if flagposzero:
        # print(len(pos[0]),len(neg[0]),overlap.max())
        # pos, pos_num = select(pos, POS_NUM)
        # pos_num = len(pos)
        # neg, neg_num = select(neg, TOTAL_NUM - POS_NUM)

        numpos = pos[0].shape[0]
        numnegall = neg[0].shape[0]
        numnegsel = min(numnegall, int(numpos * 3))
        neg, neg_num = select(neg, numnegsel)

        cls[pos] = 1

        # delta_weight[pos] = 1. / (pos_num + 1e-6)

        cls[neg] = 0
        return cls, delta, delta_weight, overlap, Counter(pos[0])


def getcropimg(img,crop_roi,box):
    xminbox,yminbox,xmaxbox,ymaxbox = box
    imageh,imagew = img.shape[:2]
    cpxmin, cpymin, cpxmax, cpymax = map(int, crop_roi)
    padleft = 0 if cpxmin > 0 else -cpxmin
    padtop = 0 if cpymin > 0 else -cpymin
    padright = 0 if cpxmax < imagew else cpxmax - imagew
    padbottom = 0 if cpymax < imageh else cpymax - imageh
    xmin = max(0, cpxmin)
    ymin = max(0, cpymin)
    xmax = min(cpxmax, imagew)
    ymax = min(cpymax, imageh)
    imgcrop = img[ymin:ymax, xmin:xmax, :]
    xminbox -= xmin
    yminbox -= ymin
    xmaxbox -= xmin
    ymaxbox -= ymin

    xminbox += padleft
    yminbox += padtop
    xmaxbox += padleft
    ymaxbox += padtop

    imgcrop = cv2.copyMakeBorder(imgcrop, padtop, padbottom, padleft, padright, cv2.BORDER_CONSTANT,
                                 value=(117,117,117))
    return imgcrop,[xminbox,yminbox,xmaxbox,ymaxbox]

def calmatchnum(RATIOS, SCALES):
    OUTPUT_SIZE = 19
    anchor_target = AnchorTarget(RATIOS, SCALES, OUTPUT_SIZE=OUTPUT_SIZE)
    anchor_match_num_dict = Counter({i: 0 for i in range(len(RATIOS) * len(SCALES))})
    N = 100000
    bbox = [80, 80, 160, 160]
    img = np.random.random((320, 320, 3))

    for _ in tqdm.tqdm(range(N)):
        xmin, ymin, xmax, ymax = map(int, bbox)
        # x1, y1, x2, y2,_ = shift_bbox_remo2([xmin, ymin, xmax, ymax])
        x1, y1, x2, y2 = shift_bbox_remo7([xmin, ymin, xmax, ymax])
        imgcrop, boxcrop = getcropimg(img, [x1, y1, x2, y2], bbox)
        imghcrop, imgwcrop = imgcrop.shape[:2]
        sx = float(160) / imgwcrop
        sy = float(160) / imghcrop
        x1 = boxcrop[0] * sx
        y1 = boxcrop[1] * sy
        x2 = boxcrop[2] * sx
        y2 = boxcrop[3] * sy
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        gtbbox = [x1, y1, x2, y2]
        cls, delta, delta_weight, overlap, anchor_match_num = anchor_target(gtbbox, OUTPUT_SIZE, False)
        anchor_match_num_dict = anchor_match_num_dict + anchor_match_num

    anchor_match_num_dict = dict(anchor_match_num_dict)
    plt.bar(list(anchor_match_num_dict.keys()), anchor_match_num_dict.values(), color='g')
    plt.title(f'remo2: RATIOS:{RATIOS}, SCALES:{SCALES}', fontsize=18)
    for i in anchor_match_num_dict:
        plt.text(i, anchor_match_num_dict.get(i) + 1, anchor_match_num_dict.get(i), ha='center', va='bottom')
    plt.show()

if __name__ == '__main__':
    # # 1. show gt w, h, area...
    # imshow_w_h_area(100000)
    # # 2. show anchor w, h
    RATIOS = [0.9, 1.2]
    SCALES = [8.5, 11, 14]
    imshow_anchors(RATIOS, SCALES)
    # 3. show anchor match num
    calmatchnum(RATIOS, SCALES)