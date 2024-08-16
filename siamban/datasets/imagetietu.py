import cv2
from PIL import  Image
import matplotlib.pyplot as plt
import numpy as np
import random
imgpath1 = '/home/lbycdy/datasets/DogCatHorse/cat_release_20220916/pic/a3549ffb8a964d5e84fb4f0a8354dfd31a28a1ff.jpg'
anno1 = [47, 283, 1453, 916]
imgpath2 = '/home/lbycdy/datasets/DogCatHorse/cat_release_20220916/pic/fc8e238deb3a41f8a53310175952ee9be1c567a7.jpg'
anno2 = [128, 76, 533, 430]
img1 = cv2.imread(imgpath1)
# cv2.rectangle(img1,(int(anno1[0]),int(anno1[1])),(int(anno1[2]),int(anno1[3])),(0,124,234),3)
img2 = cv2.imread(imgpath2)
img1_temp = img1[anno1[1]:anno1[3],anno1[0]:anno1[2]]
img1_temp2 = img1_temp.copy()
img2_temp = img2[anno2[1]:anno2[3],anno2[0]:anno2[2]]


img1[anno2[1]:anno2[3],anno2[0]:anno2[2],:] = img2_temp
img1[anno1[1]:anno1[3],anno1[0]:anno1[2],:] = img1_temp2
# img1 = cv2.add(img1[anno2[1]:anno2[3],anno2[0]:anno2[2]],img2_temp)
cv2.imshow('2221',img1)
key = cv2.waitKey(0)
if key == 27:
    exit()