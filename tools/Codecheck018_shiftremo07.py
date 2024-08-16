import random
from tqdm import tqdm

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
    """

    self.lambda_min_scale_ = -0.4
    self.lambda_max_scale_ = 0.4
    """
    xmin, ymin, xmax, ymax = box
    bw = xmax - xmin
    bh = ymax - ymin
    if TYPE_CONTEXTBOX == 0:
        cp_w = bw * 2  #
        cp_h = bh * 2
    elif TYPE_CONTEXTBOX == 1:
        cp_w = bw + KCONTEXTFACTOR * (bw + bh)
        cp_h = bh + KCONTEXTFACTOR * (bw + bh)

    cx_box = (xmin + xmax) / 2
    cy_box = (ymax + ymin) / 2

    s_rand = lambda_min_scale_ + random.uniform(0, 1.0) * (lambda_max_scale_ - lambda_min_scale_)
    r_rand = lambda_min_ratio_ + random.uniform(0, 1.0) * (lambda_max_ratio_ - lambda_min_ratio_)
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
        if coveA(box, [x1, y1, x2, y2]) > cova_thresh_shift_:
            flagok = True
            break
    if not flagok:
        x1 = cx_box - cp_w / 2
        y1 = cy_box - cp_h / 2
        x2 = x1 + cp_w
        y2 = y1 + cp_h
    return [x1, y1, x2, y2],flagok

def shift_bbox_remo7(box):  # box = [xmin,ymin,xmax,ymax]
    """

    self.lambda_min_scale_ = -0.4
    self.lambda_max_scale_ = 0.4
    """
    xmin, ymin, xmax, ymax = box
    bw = xmax - xmin
    bh = ymax - ymin
    if TYPE_CONTEXTBOX == 0:
        cp_w = bw * 2  #
        cp_h = bh * 2
    elif TYPE_CONTEXTBOX == 1:
        cp_w = bw + KCONTEXTFACTOR * (bw + bh)
        cp_h = bh + KCONTEXTFACTOR * (bw + bh)

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
        if coveA(box, [x1, y1, x2, y2]) > cova_thresh_shift_:
            flagok = True
            break
    if not flagok:
        x1 = cx_box - cp_w / 2
        y1 = cy_box - cp_h / 2
        x2 = x1 + cp_w
        y2 = y1 + cp_h
    return [x1, y1, x2, y2],flagok


TYPE_CONTEXTBOX = 1
KCONTEXTFACTOR = 0.5
lambda_min_scale_ = -0.75
lambda_max_scale_ = 0.5
lambda_min_ratio_ = -0.3
lambda_max_ratio_ = 0.3
cova_thresh_shift_ = 0.35
box = [20,20,80,80]
N = 100000
cntok = 0
for i in tqdm(range(N)):
    boxnew,flagok = shift_bbox_remo7(box)
    # boxnew,flagok = shift_bbox_remo(box)

    if flagok:
        cntok += 1
    else:
        x1,y1,x2,y2 = boxnew
        w = x2 - x1
        h = y2 - y1
print(cntok,float(cntok)/N)



