import json
import os.path
import numpy as np
from sys import platform
import cv2
from siamban.utils.bbox import corner2center, \
        Center, center2corner, Corner

path_format = '{}.{}.{}.jpg'
anno = '/home/lbycdy/datasets/COW/make_datasets.json'
cur_path = '/home/lbycdy/datasets/COW/crop511'
data = json.load(open(anno))
bbox = [0,0,0,0]


def get_bbox(image, shape):
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
for video in data:
    for track in data[video]:
        for frame in data[video][track]:
            box = data[video][track][frame]

            box[0] = int(511 * box[0])
            box[1] = int(511 * box[1])
            box[2] = int(511 * box[2] + box[0])
            box[3] = int(511 * box[3] + box[1])
            imgpath = os.path.join(cur_path,video, path_format.format(frame, track, 'x'))
            image = cv2.imread(imgpath)
            box = get_bbox(image,box)
            bbox[0] = int(box.x1)
            bbox[1] = int(box.y1)
            bbox[2] = int(box.x2)
            bbox[3] = int(box.y2)
            if image is not None:
                print(bbox)
                cv2.rectangle(image, (bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,255,0), 3)
                cv2.imshow("datasets_check",image)
                key = cv2.waitKey(0)
                if key == 27:
                    exit()
# for track in data['train2017']:
#     for frames in data['train2017'][track]:
#         anno = data['train2017'][track][frames]
#         img_path = os.path.join(cur_path,'train2017',frames)
#         image = cv2.imread(img_path)
#
#         if image is not None:
#             print(anno)
#             cv2.rectangle(image, (anno[0], anno[1]),(anno[2], anno[3]),(0,255,0), 3)
#             cv2.imshow("datasets_check",image)
#             key = cv2.waitKey(0)
#             if key == 27:
#                 exit()


# path_format = '{}.jpg'
# anno = '/home/lbycdy/datasets/OCean/GOT10K/train.json'
# cur_path = '/home/lbycdy/datasets/OCean/GOT10K/crop511'
# data = json.load(open(anno))
# bbox = [0,0,0,0]
# for imgpath in data:
#     for track in data[imgpath]:
#         for frames in data[imgpath][track]:
#             anno = data[imgpath][track][frames]
#             img_path = os.path.join(cur_path,imgpath+'/',frames+'.'+track+'.'+'x.jpg')
#             image = cv2.imread(img_path)
#
#             if image is not None:
#                 print(anno)
#                 cv2.rectangle(image, (anno[0], anno[1]),(anno[2], anno[3]),(0,255,0), 3)
#                 cv2.imshow("datasets_check",image)
#                 key = cv2.waitKey(0)
#                 if key == 27:
#                     exit()