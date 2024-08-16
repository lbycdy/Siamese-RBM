import json

jsonfile = '/home/lbycdy/datasets/DogCatHorse/AIC_ForPersonTK_20221212.json'
with open(jsonfile) as fp:
    info = json.load(fp)
for imgname in info:
    for bbox in info[imgname]:
        x1, y1, x2, y2 = bbox
