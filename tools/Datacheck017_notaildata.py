import json
import cv2
import os
import random
file = "/home/ethan/ccdimgname.txt"
fhlines = open(file).readlines()
imgnamelists = [line.strip() for line in fhlines]
print(len(imgnamelists))
imgnamelists = list(set(imgnamelists))
print(len(imgnamelists))
vid2imgname_dict = {}
for line in fhlines:
    imgname = line.strip()
    vid = "_".join(imgname.split("_")[:-1])
    try:
        vid2imgname_dict[vid].append(imgname)
    except:
        vid2imgname_dict[vid] = [imgname,]

print(len(vid2imgname_dict))
exit()
imgroot = "/mnt/data/GOT/CatDogHorse"
jsonlist = [
    "/mnt/DataHDD/Datasets/GOT/CatDogHorse/Layout/REMOWIKICATHIGDOGANNORESULT20230427.json",
    "/mnt/DataHDD/Datasets/GOT/CatDogHorse/Layout/REMOWIKICATHIGDOGANNORESULT20230504.json",
    "/mnt/DataHDD/Datasets/GOT/CatDogHorse/Layout/REMOWIKICATHIGDOGANNORESULT20230508.json"
]
"""
{1: 168138, 2: 100862, 3: 74215,}
"""
jsonlist = [
    "/mnt/DataHDD/Datasets/GOT/CatDogHorse/Layout/REMOWIKICATHIGDOGANNORESULT20230526.json",
    "/mnt/DataHDD/Datasets/GOT/CatDogHorse/Layout/REMOWIKICATHIGDOGANNORESULT20230525.json",
]
"""
{1: 201568,2: 134581, 3: 108643 }
"""
colorlist = [(0,0,255),(255,0,0),(0,255,0)]
dataall = {}
for file in jsonlist:
    data=json.load(open(file))
    for imgpath in data:
        dataall[imgpath]=data[imgpath]
cnt = 0
for imgpath in dataall:
    imgname = os.path.basename(imgpath)
    if "C_"==imgname[:2] or "CD"==imgname[:2] or "D_"==imgname[:2]:
        cnt+=1
print(cnt,len(dataall),len(dataall)-cnt)
"""
"/mnt/DataHDD/Datasets/GOT/CatDogHorse/Layout/REMOWIKICATHIGDOGANNORESULT20230526.json",
"/mnt/DataHDD/Datasets/GOT/CatDogHorse/Layout/REMOWIKICATHIGDOGANNORESULT20230525.json",
91474 319889 228415
"""

cid2cnt_CCD_dict = {}
cid2cnt_notCCD_dict = {}
cid2cnt_dict = {}

for imgpath in dataall:
    imgname = os.path.basename(imgpath)
    if "C_" == imgname[:2] or "CD" == imgname[:2] or "D_" == imgname[:2]:
        for box in dataall[imgpath]:
            xmin,ymin,xmax,ymax,cid = map(int,box)
            try:
                cid2cnt_CCD_dict[cid]+=1
            except:
                cid2cnt_CCD_dict[cid]=1
    else:
        for box in dataall[imgpath]:
            xmin,ymin,xmax,ymax,cid = map(int,box)
            try:
                cid2cnt_notCCD_dict[cid]+=1
            except:
                cid2cnt_notCCD_dict[cid]=1
    for box in dataall[imgpath]:
        xmin, ymin, xmax, ymax, cid = map(int, box)
        try:
            cid2cnt_dict[cid] += 1
        except:
            cid2cnt_dict[cid] = 1
print(cid2cnt_dict)
print(cid2cnt_CCD_dict)
print(cid2cnt_notCCD_dict)
"""
"/mnt/DataHDD/Datasets/GOT/CatDogHorse/Layout/REMOWIKICATHIGDOGANNORESULT20230526.json",
"/mnt/DataHDD/Datasets/GOT/CatDogHorse/Layout/REMOWIKICATHIGDOGANNORESULT20230525.json",
cid2cnt_dict：        {1: 201568, 2: 134581, 3: 108643}
cid2cnt_CCD_dict：    {1: 107850, 2: 16203}
cid2cnt_notCCD_dict： {1: 93718,  2: 118378, 3: 108643}
"""
fh = open("ccdimgname.txt","w")
for imgpath in dataall:
    imgname = os.path.basename(imgpath)
    if "C_" == imgname[:2] or "CD" == imgname[:2] or "D_" == imgname[:2]:
        fh.writelines("%s\n"%imgname)
fh.close()

#
# for cid in cid2imgpath_dict:
#     imgboxlist = cid2imgpath_dict[cid]
#     random.shuffle(imgboxlist)
#     for imgpath,boxes in imgboxlist:
#         if "CatDog_dataset_221117" in imgpath:
#             img = cv2.imread(os.path.join(imgroot, imgpath))
#             imgh, imgw = img.shape[:2]
#             for box in boxes:
#                 xmin, ymin, xmax, ymax, cid = map(int, box)
#                 cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colorlist[cid-1], int(imgh / 200))
#             cv2.namedWindow("img", cv2.NORM_MINMAX)
#             cv2.imshow("img", img)
#             key = cv2.waitKey()
#             if key == 27:
#                 exit()
#             elif key==ord("n"):
#                 break

# for file in jsonlist:
#     data=json.load(open(file))
#     for imgpath in data:
#         img = cv2.imread(os.path.join(imgroot,imgpath))
#         imgh,imgw = img.shape[:2]
#         for box in data[imgpath]:
#             xmin,ymin,xmax,ymax,cid = map(int,box)
#             cv2.rectangle(img,(xmin,ymin),(xmax,ymax),colorlist[cid],int(imgh/200))
#         cv2.namedWindow("img",cv2.NORM_MINMAX)
#         cv2.imshow("img",img)
#         key=cv2.waitKey()
#         if key==27:
#             exit()


