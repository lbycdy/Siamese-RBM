import torch
import torch.nn.functional as F
from natsort import natsorted
import os
keylist = ["head.cls_weight","head.loc_weight"]
snapshot_folder = '/media/ethan/OldDisk/home/ethan/Models/Results/0016_GOT/20221101V001GOTMobileOne_CorrAndWeightAddC128_rankloss_4GPU127siambanacm'
filelist = os.listdir(snapshot_folder)
filelist = natsorted(filelist)
for key in keylist:
    print(key)
    for file in filelist:
        param = torch.load(os.path.join(snapshot_folder,file), map_location=torch.device('cpu'))
        print(F.softmax(param[key]),file)

