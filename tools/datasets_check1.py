import json
import os.path
from sys import platform
import cv2

path_format = '{}.jpg'
anno = '/home/lbycdy/datasets/DogCatHorse/AIC_ForPersonTK_20221212.json'
cur_path = '/home/lbycdy/datasets/DogCatHorse/'
data = json.load(open(anno))
bbox = [0,0,0,0]
for imgpath in data:
    for box in data[imgpath]:
        img_path = os.path.join(cur_path, imgpath )
        image = cv2.imread(img_path)
        if image is not None:
            print(bbox)
            cv2.rectangle(image, (box[0], box[1]),(box[2], box[3]),(0,255,0), 3)
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