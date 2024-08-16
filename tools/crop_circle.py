import cv2
import numpy as np
img = cv2.imread('/home/lbycdy/work/自然场景下奶牛单目标跟踪方法/微信图片_20230405172710.png')
hh, ww = img.shape[:2]
radius1 = 75
radius2 = 75
xc = hh // 2
yc = ww // 2
mask1 = np.zeros_like(img)
mask1 = cv2.circle(mask1, (xc,yc), radius1, (255,255,255), -1)
mask2 = np.zeros_like(img)
# mask2 = cv2.circle(mask2, (xc,yc), radius2, (255,255,255), -1)
# mask = cv2.subtract(mask2, mask1)
result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask1[:,:,0]
cv2.imwrite('/home/lbycdy/work/自然场景下奶牛单目标跟踪方法/lens.png', result)