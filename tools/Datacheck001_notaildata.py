import json
import cv2
import os
import random
jsonlist = [
    "/home/lbycdy/cdh_json_new_230524/REMOWIKICATHIGDOGANNORESULT20230525.json",
    "/home/lbycdy/cdh_json_new_230524/REMOWIKICATHIGDOGANNORESULT20230526.json",
]
video_temp = ''
dataall = {}
imgpath_list = []
box_list = []
videolist = []
cat_list = []
dog_list = []
horse_list = []
other_list = []
cat_img_list = []
dog_img_list = []
horse_img_list = []
cat_vid_list = []
dog_vid_list = []
horse_vid_list = []
for file in jsonlist:
    data=json.load(open(file))
    for imgpath in data:
        dataall[imgpath]=data[imgpath]
for imgpath in dataall:
    # print(imgpath)
    imgname = os.path.basename(imgpath)

    if "C_"==imgname[:2] or "CD"==imgname[:2] or "D_"==imgname[:2]:

        a = imgpath.split('/')
        b = a[:-1]
        # print(b)
        if len(b) == 2:
            video = os.path.join(b[0],b[1])
        else:
            # print(111)
            video = b[0]
        if video not in videolist:
            videolist.append(video)
print(videolist)
path = '/home/lbycdy/cdh_json_new_230524/REMOWIKICATHIGDOGANNORESULTVID20230522711.json'
with open(path,"w+",encoding="utf-8") as f:
    f.write("{")
    for video1 in videolist:
        f.write("\n")
        f.write("    ")
        f.write('"')
        f.write("%s" % video1)
        f.write('"')
        f.write(":{\n")
        f.write("        ")
        f.write('"00": {\n')

        for imgpath in dataall:
            imgname = os.path.basename(imgpath)
            if "C_" == imgname[:2] or "CD" == imgname[:2] or "D_" == imgname[:2]:

                box = dataall[imgpath]

                if len(box)==1:
                    # print(box[0][4])
                    f.write("            ")
                    a = imgpath.split('/')
                    imgname = a[-1]
                    f.write('"')
                    f.write(imgname)
                    f.write('": [\n')

                    f.write("                ")
                    f.write(str(box[0][0]))
                    f.write(",\n")
                    f.write("                ")
                    f.write(str(box[0][1]))
                    f.write(",\n")
                    f.write("                ")
                    f.write(str(box[0][2]))
                    f.write(",\n")
                    f.write("                ")
                    f.write(str(box[0][3]))
                    f.write("\n")
                    f.write("            ")
                    f.write("],")
                    f.write("\n")
                    # f.write("        ")