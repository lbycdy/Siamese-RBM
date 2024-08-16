import random
import math
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import cv2
from siamban.utils.bbox import corner2center, \
        Center, center2corner, Corner
import cv2
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
        ori = im_c - size //2 * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((2, size, size), dtype=np.float32)
        points[0, :, :], points[1, :, :] = x.astype(np.float32), y.astype(np.float32)
        return points

p = Point(8,19,160/2)
print(p.points.min(),p.points.max())
#S8: 40-120
#S9: 35-125
#S10: 30-130
#S11: 25-135
#S12: 20-140

# exit()
class PointTarget1:
    def __init__(self, stride,outputsize,imgcenter):
        self.points = Point(stride,outputsize, imgcenter)#160/11=14.5

    def __call__(self, target, size, neg=False):
        SCALE_POS = 4.0
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
        pos = np.where(np.square(tcx - points[0]) / np.square(tw / SCALE_POS) +
                       np.square(tcy - points[1]) / np.square(th / SCALE_POS) < 1)

        pos_num = len(pos[0])
        delta_weight[0,pos] = 1. / (pos_num + 1e-6)
        cls[pos] = 1

        return cls, delta,delta_weight



class PointTarget2_Rect:
    def __init__(self, stride,scalepos):
        self.points = Point(stride,11, 80)#160/11=14.5
        self.SCALE_POS = scalepos

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

        flag1 = np.abs(tcx-points[0])<tw / self.SCALE_POS
        flag2 = np.abs(tcy-points[1])<th / self.SCALE_POS
        flag = np.bitwise_and(flag1,flag2)
        pos = np.where(flag)
        pos_num = len(pos[0])
        delta_weight[0,pos] = 1. / (pos_num + 1e-6)
        cls[pos] = 1

        return cls, delta,delta_weight


class PointTarget3_Square:
    def __init__(self, ):
        self.points = Point(14,11, 80)#160/11=14.5

    def __call__(self, target, size, neg=False):
        SCALE_POS = 4.0
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
        sizemean = min(tw,th)
        flag1 = np.abs(tcx-points[0])<(sizemean / SCALE_POS)
        flag2 = np.abs(tcy-points[1])<(sizemean / SCALE_POS)
        flag = np.bitwise_and(flag1,flag2)
        pos = np.where(flag)
        pos_num = len(pos[0])
        delta_weight[0,pos] = 1. / (pos_num + 1e-6)
        cls[pos] = 1

        return cls, delta,delta_weight


class PointTarget3_SquareMin:
    def __init__(self, stride,scalepos):
        self.points = Point(stride,11, 80)#160/11=14.5
        self.SCALE_POS = scalepos
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
        sizemean = min(tw,th)
        flag1 = np.abs(tcx-points[0])<sizemean / self.SCALE_POS
        flag2 = np.abs(tcy-points[1])<sizemean / self.SCALE_POS
        flag = np.bitwise_and(flag1,flag2)
        pos = np.where(flag)
        pos_num = len(pos[0])
        delta_weight[0,pos] = 1. / (pos_num + 1e-6)
        cls[pos] = 1

        return cls, delta,delta_weight

shift_motion_model = True
kContextFactorShiftBox = 2.0
def sample_exp_two_sided(lam):
    prob = random.random()
    if prob > 0.5:
        pos_or_neg = 1
    else:
        pos_or_neg = -1
    rand_uniform = random.random()
    return math.log(rand_uniform) / lam * pos_or_neg
def shift_bbox(image, box,lambda_shift_,lambda_scale_,lambda_min_scale_=-0.4,lambda_max_scale_=0.4):  # box = [xmin,ymin,xmax,ymax]
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
            width_scale_factor = lambda_min_scale_ + rand_num*(lambda_max_scale_ - lambda_min_scale_)
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
            height_scale_factor = lambda_min_scale_ + rand_num*(
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
        flagok = False
    else:
        x1 = new_center_x - new_width / 2
        x2 = new_center_x + new_width / 2
        y1 = new_center_y - new_height / 2
        y2 = new_center_y + new_height / 2
        flagok=True
    w = (x2 - x1)*2
    h = (y2 - y1)*2
    xc = (x1+x2)/2
    yc = (y1+y2)/2
    x1 = xc - w/2
    x2 = xc + w/2
    y1 = yc - h/2
    y2 = yc + h/2
    return x1, y1, x2, y2,flagok

def coveA(box1, box2):
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

def shift_bbox_remo(box):  # box = [xmin,ymin,xmax,ymax]
    xmin,ymin,xmax,ymax = box
    cp_w = (xmax - xmin)*2
    cp_h = (ymax - ymin)*2
    cx_box = (xmin + xmax)/2
    cy_box = (ymax + ymin)/2
    scalemin = -0.4
    scalemax = 0.4
    sx = scalemin+random.uniform(0,1.0)*(scalemax - scalemin)
    sy = scalemin+random.uniform(0,1.0)*(scalemax - scalemin)
    cp_w *= (sx + 1.0)
    cp_h *= (sy + 1.0)


    shiftleft = cx_box - cp_w
    shiftright = cx_box
    shifttop = cy_box - cp_h
    shiftbottom = cy_box
    shiftleft, shiftright,shifttop,shiftbottom = map(int,[shiftleft, shiftright,shifttop,shiftbottom])
    flagok = False
    for i in range(20):
        if shiftleft<shiftright:
            x1 = random.randint(shiftleft,shiftright)
        else:
            x1 = xmin
        if shifttop<shiftbottom:
            y1 = random.randint(shifttop,shiftbottom)
        else:
            y1 = ymin
        x2 = x1 + cp_w
        y2 = y1 + cp_h
        if coveA(box,[x1,y1,x2,y2])>0.8:
            flagok = True
            break
    if not flagok:
        x1 = cx_box - cp_w/2
        y1 = cy_box - cp_h/2
        x2 = x1 + cp_w
        y2 = y1 + cp_h
    return x1,y1,x2,y2,True

def shift_bbox_remo_modify(box):  # box = [xmin,ymin,xmax,ymax]
    """

    self.lambda_min_scale_ = -0.4
    self.lambda_max_scale_ = 0.4
    """
    lambda_min_scale_ = -0.5
    lambda_max_scale_ = 0.25
    lambda_min_ratio_ = -0.4
    lambda_max_ratio_ = 0.4
    cova_emit_ = 0.8
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
        if coveA(box, [x1, y1, x2, y2]) > cova_emit_:
            flagok = True
            break
    if not flagok:
        x1 = cx_box - cp_w / 2
        y1 = cy_box - cp_h / 2
        x2 = x1 + cp_w
        y2 = y1 + cp_h
    return x1, y1, x2, y2,flagok

def shift_bbox_remo2(box):  # box = [xmin,ymin,xmax,ymax]
    """

    self.lambda_min_scale_ = -0.4
    self.lambda_max_scale_ = 0.4
    """
    lambda_min_scale_ = -0.5
    lambda_max_scale_ = 0.25
    lambda_min_ratio_ = -0.4
    lambda_max_ratio_ = 0.4
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
    cp_w = max(cp_w, bw)
    cp_h = max(cp_h, bh)

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
    return x1, y1, x2, y2,True


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

def coveA(box1, box2):
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

file = '/home/lbycdy/REMOWIKICATHIGDOGANNORESULT20230504.json'
imgroot = '/home/lbycdy/datasets/DogCatHorse'
fhlines = open(file).readlines()
data = json.load(open(file))
netimgsize = 160


point_target1_s16out13 = PointTarget1(16,13,netimgsize/2)
point_target1_s8out19 = PointTarget1(8,19,80)

# point_target2_rect10_S4 = PointTarget2_Rect(10,4)
# point_target3_square = PointTarget3_Square()
# point_target3_squaremins8_S4 = PointTarget3_SquareMin(stride=8,scalepos=4)
# point_target3_squaremins10_S4 = PointTarget3_SquareMin(stride=10,scalepos=4)
# point_target3_squaremins10_S3_5 = PointTarget3_SquareMin(stride=10,scalepos=3.5)
# point_target3_squaremins14_S4 = PointTarget3_SquareMin(stride=14,scalepos=4)

for imgpath in data:
    for idxbox, box in enumerate(data[imgpath]):
        print("-----------------------------------")
        imgname = imgpath
        xmin,ymin,xmax,ymax,cid = map(int,box)
        if cid==1:
            wbox = xmax - xmin
            hbox = ymax -ymin
            rorg = float(wbox)/hbox
            box = [xmin,ymin,xmax,ymax]
            img = cv2.imread(os.path.join(imgroot,imgname))

            x1,y1,x2,y2,flagok = shift_bbox_remo(box)
            # x1,y1,x2,y2,flagok  = shift_bbox(img, box, lambda_shift_=5.0, lambda_scale_=15.0, lambda_min_scale_=-0.4, lambda_max_scale_=0.4)
            imgcrop,boxcrop = getcropimg(img,[x1,y1,x2,y2],box)
            imghcrop,imgwcrop = imgcrop.shape[:2]
            sx = float(netimgsize)/imgwcrop
            sy = float(netimgsize)/imghcrop
            x1 = boxcrop[0] * sx
            y1 = boxcrop[1] * sy
            x2 = boxcrop[2] * sx
            y2 = boxcrop[3] * sy
            x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
            wbox = x2 - x1
            hbox = y2 - y1
            r = float(wbox) / hbox
            imgcrop160 = cv2.resize(imgcrop,(netimgsize,netimgsize))
            imgcrop160show = imgcrop160.copy()
            cv2.rectangle(imgcrop160show,(x1,y1),(x2,y2),(0,0,255),1)
            cv2.putText(imgcrop160show,"%.02f-%.02f"%(rorg,r),(0,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))
            cv2.namedWindow("imgcropresm0_160", cv2.NORM_HAMMING)
            cv2.imshow("imgcropresm0_160", imgcrop160show)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(netimgsize, x2)
            y2 = min(netimgsize, y2)

            bbox = [x1, y1, x2, y2]
            outsize = 13
            cls, delta, delta_weight = point_target1_s16out13(bbox, outsize, False)
            clshow = cls * 255
            clshow = clshow.astype(np.uint8)
            cv2.namedWindow("clsorgs8out11", cv2.NORM_HAMMING)
            cv2.imshow("clsorgs8out11", clshow)
            imgcrop160show = imgcrop160.copy()
            points = point_target1_s16out13.points.points
            for i in range(outsize):
                for j in range(outsize):
                    cid = cls[i,j]
                    x,y = points[:,i,j]
                    x,y = int(x),int(y)
                    if cid==1:
                        cv2.circle(imgcrop160show,(x,y),3,(0,0,255),-1)
                    else:
                        cv2.circle(imgcrop160show,(x,y),3,(0,255,0),-1)
            cv2.namedWindow("clsorgs8out11img", cv2.NORM_HAMMING)
            cv2.imshow("clsorgs8out11img", imgcrop160show)

            cls, delta, delta_weight = point_target1_s8out19(bbox, 19, False)
            clshow = cls * 255
            clshow = clshow.astype(np.uint8)
            cv2.namedWindow("clsorgs8out19", cv2.NORM_HAMMING)
            cv2.imshow("clsorgs8out19", clshow)
            imgcrop160show = imgcrop160.copy()
            points = point_target1_s8out19.points.points
            print(points.min(),points.max())
            for i in range(19):
                for j in range(19):
                    cid = cls[i, j]
                    x, y = points[:, i, j]
                    x, y = int(x), int(y)
                    if cid == 1:
                        cv2.circle(imgcrop160show, (x, y), 3, (0, 0, 255), -1)
                    else:
                        cv2.circle(imgcrop160show, (x, y), 3, (0, 255, 0), -1)
            cv2.namedWindow("clsorgs8out19img", cv2.NORM_HAMMING)
            cv2.imshow("clsorgs8out19img", imgcrop160show)
            key = cv2.waitKey()
            if key==27:
                exit()



