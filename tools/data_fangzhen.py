import numpy as np
import random
import math
import cv2
from siamban.utils.bbox import corner2center, \
        Center, center2corner, Corner

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
def shift_bbox_remo7(box):  # box = [xmin,ymin,xmax,ymax]
    """

    self.lambda_min_scale_ = -0.4
    self.lambda_max_scale_ = 0.4
    """
    xmin, ymin, xmax, ymax = box
    bw = xmax - xmin
    bh = ymax - ymin

    cp_w = bw + 0.5 * (bw + bh)
    cp_h = bh + 0.5 * (bw + bh)

    cx_box = (xmin + xmax) / 2
    cy_box = (ymax + ymin) / 2

    s_rand = -0.5 + random.uniform(0, 1.0) * 1.0
    r_rand = -0.5 + random.uniform(0, 1.0) * 1.0
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



img = np.random.random((224,224,3))
box = [40,40,120,120]
xmin_list = []
ymin_list = []
xmax_list = []
ymax_list = []
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

for i in range(1000):
    x1, y1, x2, y2 = box

    x1_crop = max(0, x1)
    y1_crop = max(0, y1)
    x2_crop = min(x2, 160)
    y2_crop = min(y2, 160)
    bbox_crop = Corner(x1_crop, y1_crop, x2_crop, y2_crop)
    img[x1_crop:x2_crop,y1_crop:y2_crop,:] = RandomErasing(img[x1_crop:x2_crop,y1_crop:y2_crop,:],0.02,0.4,0.3)
    bbox = Corner(x1, y1, x2, y2)

    img = img.astype(np.uint8).copy()
    # cv2.rectangle(img, (x1_crop, y1_crop), (x2_crop, y2_crop), (0, 255, 0))

    cv2.imshow("imgs",img)

    key = cv2.waitKey()
    if key==27:
        exit()