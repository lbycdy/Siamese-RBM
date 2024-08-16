import json
import os.path
from os.path import join, isdir
from sys import platform
from siamban.utils.bbox import center2corner, Center
import cv2
import math
import numpy as np
from collections import namedtuple
import random

anno = '/home/lbycdy/datasets/COW/make_datasets.json'
path='/home/lbycdy/precision plot/9.txt'
data = json.load(open(anno))
bbox = [0,0,0,0]
file = open(path, 'w+')
lambda_min_ratio_ = -0.9
lambda_max_ratio_ = 0.5
lambda_min_scale_ = -0.0
lambda_max_scale_ = -0.0
lambda_scale_ = 10.0

def sample_exp_two_sided(lam):
    prob = random.random()
    if prob > 0.5:
        pos_or_neg = 1
    else:
        pos_or_neg = -1
    rand_uniform = random.random()
    return math.log(rand_uniform) / lam * pos_or_neg
for imgpath in data:

    for track in data[imgpath]:
        frames = data[imgpath][track]

        for box in data[imgpath][track]:
            anno = data[imgpath][track][box]
            r_rand = lambda_min_ratio_ + random.uniform(0, 1.0) * (lambda_max_ratio_ - lambda_min_ratio_)
            smp_d = sample_exp_two_sided(lambda_scale_)
            s_rand = max(lambda_min_scale_, min(lambda_max_scale_, smp_d))
            xc, yc, w, h = anno
            w *= (1.0 + r_rand)
            h /= (1.0 + r_rand)
            w *= (1.0 + s_rand)
            h *= (1.0 + s_rand)
            anno[0] = xc-w/2.0
            anno[1] = yc-h/2.0
            anno[2] = xc+w/2.0
            anno[3] = yc+h/2.0
            for i in range(3):
                file.write(str(int(anno[i]*511)))
                file.write(', ')
            file.write(str(int(anno[3] * 511)))
            file.write('\n')
            print(anno)

