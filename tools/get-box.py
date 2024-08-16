import json
import os.path
from os.path import join, isdir
from sys import platform
from siamban.utils.bbox import center2corner, Center
import cv2
import numpy as np
from collections import namedtuple

anno = '/home/lbycdy/datasets/COW/make_datasets.json'
path='/home/lbycdy/113.txt'
data = json.load(open(anno))
bbox = [0,0,0,0]
file = open(path, 'w+')
for imgpath in data:

    for track in data[imgpath]:
        frames = data[imgpath][track]

        for box in data[imgpath][track]:
            anno = data[imgpath][track][box]
            xc, yc, w, h = anno
            anno[0] = xc - w / 2.0
            anno[1] = yc - h / 2.0
            anno[2] = xc + w / 2.0
            anno[3] = yc + h / 2.0
            for i in range(3):
                file.write(str(int(anno[i]*511)))
                file.write(', ')
            file.write(str(int(anno[3] * 511)))
            file.write('\n')
            print(anno)

