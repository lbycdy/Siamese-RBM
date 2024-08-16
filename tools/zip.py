import cv2
import os
import math
import  glob
import shutil
from tqdm import  tqdm
imgroot = "/home/lbycdy/datasets/OCean/GOT10K/crop511-anno"
imgrootdst = imgroot + "_zip"
if not os.path.exists(imgrootdst):
    os.makedirs(imgrootdst)
# imglists = os.listdir(imgroot)
imglists = []
imglists.sort()
# for file in glob.glob(imgroot+'/train/'+'*/*.jpg'):
#
#     imglists.append(file)
imglists.sort()
cntfolder = 0
cntimg = 0
imgperfolder = 50
folder = os.path.join(imgrootdst,"%03d"%cntfolder)
if not os.path.exists(folder):
    os.makedirs(folder)
for imgname in tqdm(imglists):
    imgpathsrc = os.path.join(imgroot, imgname)
    shutil.copy2(imgpathsrc,folder)
    cntimg += 1
    if cntimg>imgperfolder:
        cntimg = 0
        cntfolder += 1
        folder = os.path.join(imgrootdst, "%03d" % cntfolder)
        os.makedirs(folder)
