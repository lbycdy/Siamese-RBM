import torch
from torch import nn
nn.BatchNorm2d


snapshot1 = "/home/lbycdy/checkpoint_e19_param.pth"
snapshot2 = "/home/lbycdy/checkpoint_e1_param.pth"
snapshot3 = "/home/lbycdy/checkpoint_e0_param.pth"

param1 = torch.load(snapshot1,map_location=torch.device('cpu'))
param2 = torch.load(snapshot2, map_location=torch.device('cpu'))
# param3 = torch.load(snapshot3, map_location=torch.device('cpu'))
for key1 in param1:
    print("11111111",key1,param1[key1].min(),param1[key1].max())
    print("222222222",key1,param2[key1].min(),param2[key1].max())
    # print("333333333", key1, param3[key1].min(), param3[key1].max())



# head.loc.4 tensor([-0.0261,  0.0085, -0.0485, -0.1621,  0.2001, -0.1582,  0.2481, -0.3447,
#         -0.2976, -0.1020, -0.1125,  0.1047, -0.1045, -0.0324, -0.0126, -0.1002,
#         -0.2987, -0.3418,  0.0680,  0.1690,  0.0361, -0.0498,  0.0097, -0.0344,
#          0.0964, -0.0614,  0.2058,  0.0040,  0.2292, -0.1359,  0.2537,  0.0236],
#        device='cuda:0')