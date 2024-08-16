import cv2
import numpy as np

root = '/home/lbycdy/work/自然场景下奶牛单目标跟踪方法/Document1_01.png'
img = cv2.imread(root)
# cv2.rectangle(img,(278,220),(1208,450),(0,0,255),1)
img1 = np.zeros((230,930),np.uint8)
img1 = img[220:450,278:1208]
cv2.imwrite('/home/lbycdy/1.jpg',img1)
cv2.imshow('ddd', img1)

key = cv2.waitKey(0)
if key == 27:
    exit()