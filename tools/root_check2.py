import json
import os
import cv2
root = '/home/zhangming/df95a216-fc35-46f2-ab9f-02e360a22c20/Datasets/GOT/CatDogHorse'
anno = '/home/zhangming/df95a216-fc35-46f2-ab9f-02e360a22c20/Datasets/GOT/CatDogHorse/Layout/train_until20221105.json'
root = '/home/lbycdy/datasets/DogCatHorse/personWithCatDogHorse'
anno = '/home/lbycdy/datasets/DogCatHorse/train_until20221201(1).json'


data = json.load(open((anno)))
for imgpath in data:

    img_path = os.path.join(root, imgpath)
    img = cv2.imread(img_path)
    cv2.imwrite("img",img)



