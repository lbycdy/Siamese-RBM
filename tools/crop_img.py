import cv2
import numpy as np
img = cv2.imread('/home/lbycdy/work/自然场景下奶牛单目标跟踪方法/微信图片_20230710104937.jpg')
img = cv2.resize(img,(150,200))
cv2.imwrite('/home/lbycdy/img.jpg',img)
# cv2.imshow('img',img)
#
# key = cv2.waitKey(0)
# if key == 27:
#     exit()