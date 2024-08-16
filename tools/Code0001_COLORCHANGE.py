import os
import cv2

root = '/home/lbycdy/datasets/1.jpg'
img = cv2.imread(root)
# print(img)
a = img.shape
num = 0
for row in range(a[0]):
    for col in range(a[1]):
        (b,g,r) = img[row,col]
        if r>=100 and b<70 and g<70:
            img[row,col] = (203,192,255)
            num+=1
# cv2.imshow('lbycdy',img)
cv2.imwrite("/home/lbycdy/datasets/img.jpg",img)
print(num)
# key = cv2.waitKey(0)
# if key == 27:
#     exit()