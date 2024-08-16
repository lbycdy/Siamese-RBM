import json
import os

root = '/home/zhangming/df95a216-fc35-46f2-ab9f-02e360a22c20/Datasets/GOT/CatDogHorse'
anno = '/home/zhangming/df95a216-fc35-46f2-ab9f-02e360a22c20/Datasets/GOT/CatDogHorse/Layout/train_until20221105.json'


data = json.load(open((anno)))
for imgpath in data:

    img_path = os.path.join(root, imgpath)

    a = os.path.isfile(img_path)
    if a is not True:
        print(a)



