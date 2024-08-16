import json
import os.path
from os.path import join, isdir
from sys import platform
from siamban.utils.bbox import center2corner, Center
import cv2
import numpy as np
from collections import namedtuple
path_format = '{}.jpg'
anno = '/home/lbycdy/datasets/OCean/YTB/train_imgallexist_withclassname.json'
cur_path = '/home/lbycdy/datasets/OCean/YTB/crop511'
cur_path1 = '/home/lbycdy/datasets/OCean/YTB/crop511-anno'
data = json.load(open(anno))
bbox = [0,0,0,0]
Corner = namedtuple('Corner', 'x1 y1 x2 y2')
def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2
def _get_bbox(image, shape):
    imh, imw = image.shape[:2]  # 图片宽高
    if len(shape) == 4:
        w, h = shape[2] - shape[0], shape[3] - shape[1]  # bb宽高
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
clstag2clsid_dict = {
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
imgpath11 = ''
for imgpath in data:
    imgpath11 = imgpath
    for track in data[imgpath]:
        frames = data[imgpath][track]
        #if name == 'LASOT':
        # c = os.path.dirname(imgpath)
        #elif name == 'GOT10K':
        # c = data[imgpath][track]['cls'][1].strip()
        #elif name == 'YOUTUBEBB':
        c = data[imgpath][track]['cls'].strip()
        #else:
        # c = data[imgpath][track]['cls']
        c = str(c)
        if c in clstag2clsid_dict:
            for box in data[imgpath][track]:
                anno = data[imgpath][track][box]


                img_path = os.path.join(cur_path,imgpath+'/',box+'.'+track+'.'+'x.jpg')
                if box != 'cls':
                    anno[0] = anno[0] * 127
                    anno[1] = anno[1] * 127
                    anno[2] = anno[2] * 127
                    anno[3] = anno[3] * 127
                    img_path1 = os.path.join(cur_path1, imgpath + '/', box + '.' + track + '.' + 'x.jpg')
                    a = imgpath11.find("/")
                    root1 = imgpath11[0:a]
                    imgpath11 = imgpath11[a+1:len(imgpath11)]
                    b = imgpath11.find("/")
                    root2 = imgpath11[0:b]
                    root3 = imgpath11[b+1:len(imgpath11)]
                    if not isdir(os.path.join(cur_path1,root1)):
                        os.mkdir(os.path.join(cur_path1,root1))
                    if not isdir(os.path.join(cur_path1, root1+'/',root2)):
                        os.mkdir(os.path.join(cur_path1, root1+'/',root2))
                    if not isdir(os.path.join(cur_path1, root1+'/',root2+'/',root3)):
                        os.mkdir(os.path.join(cur_path1, root1+'/',root2+'/',root3))
                    print(img_path)
                    image = cv2.imread(img_path)
                # cv2.imshow("datasets_check", image)
                # h = image.shape[0]
                # w = image.shape[1]
                # bbox[0] = int(anno[0] * w - anno[2] * w/2)
                # bbox[1] = int(anno[1] * h - anno[3] * h/2)
                # bbox[2] = int(anno[0] * w + anno[2] * w/2)
                # bbox[3] = int(anno[1] * h + anno[3] * h/2)
                # print(bbox)
                # cv2.rectangle(image, (bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,255,0), 3)
                    if image != None:
                        bbox = _get_bbox(image, anno)
                        anno[0] = int(bbox[0])
                        anno[1] = int(bbox[1])
                        anno[2] = int(bbox[2])
                        anno[3] = int(bbox[3])
                        print(anno)
                        cv2.rectangle(image, (anno[0], anno[1]), (anno[2], anno[3]), (0, 255, 0), 3)
                        cv2.imshow("datasets_check",image)
                        # cv2.imwrite("%s"%img_path1,image)
                        key = cv2.waitKey(0)
                        if key == 27:
                            exit()