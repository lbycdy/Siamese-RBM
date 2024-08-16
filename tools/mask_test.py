import cv2
import os

import numpy as np

root1 = '/home/lbycdy/datasets/DogCatHorse/keypoint_train_images_20170902/0002c590c6b4b5d3fcecfc3d9177756c614ca0ad.jpg'
root2 = '/home/lbycdy/datasets/DogCatHorse/AIC_Data_20230630_orisize/0002c590c6b4b5d3fcecfc3d9177756c614ca0ad#OriW940-H733-S2.042553_001person.jpg'
img = cv2.imread(root1)


img1 = cv2.imread(root2)
img2 = np.zeros_like(img)
img2[:,:,0] = img[:,:,0] * img1[:,:,0]
img2[:,:,1] = img[:,:,1] * img1[:,:,1]
img2[:,:,2] = img[:,:,2] * img1[:,:,2]
cv2.imshow('22',img2)
cv2.imshow('dd',img)
# cv2.imshow('44',img1)
key = cv2.waitKey(0)
if key == 27:
    exit()
