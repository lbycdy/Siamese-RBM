# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.rank_loss import select_cross_entropy_loss, select_iou_loss, rank_cls_loss,rank_loc_loss

from siamban.models.backbone.mobileone_stride import mobileone
from siamban.models.backbone.mobileone_strideS16OutTwo import mobileone as mobileones16outtwo

import torch
import torchvision
import math
import numpy as np
import random
class Point:
    """
    This class generate points.
    """
    def __init__(self, stride, size, image_center):
        self.stride = stride
        self.size = size
        self.image_center = image_center

        self.points = self.generate_points(self.stride, self.size, self.image_center)

    def generate_points(self, stride, size, im_c):
        ori = im_c - size // 2 * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((2, size, size), dtype=np.float32)
        points[0, :, :], points[1, :, :] = x.astype(np.float32), y.astype(np.float32)

        return points

class AdjustAllLayerFPNAddFlagCenterCrop(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayerFPNAddFlagCenterCrop, self).__init__()
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels[1], 1, bias=False),
            nn.BatchNorm2d(out_channels[1]),
        )
        self.downsample3  = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels[2], 1, bias=False),
            nn.BatchNorm2d(out_channels[2]),
        )
        self.weight = nn.Parameter(torch.ones(3))

    def forward(self, features,flagcentercrop=False):
        downsample1 = self.downsample1(features[0])
        downsample2 = self.downsample2(features[1])
        downsample3 = self.downsample3(features[2])
        weight = F.softmax(self.weight, 0)

        out = downsample1*weight[0]+downsample2*weight[1]+downsample3*weight[2]
        return out

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )

        self.xorr_search = nn.Conv2d(hidden, hidden, kernel_size=7, bias=False)
        self.xorr_kernel = nn.Conv2d(hidden, hidden, kernel_size=7, bias=True)
        self.xorr_activate = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, search,kernel):
        kernel = self.conv_kernel(kernel)
        kernel_part = self.xorr_kernel(kernel)
        search = self.conv_search(search)
        search_part = self.xorr_search(search)
        tmp = self.relu(search_part+kernel_part)
        feature = self.xorr_activate(tmp)
        return feature

class RPNHead(nn.Module):
    def __init__(self,cin = 256):
        super(RPNHead, self).__init__()
        # self.depthcorr = DepthwiseXCorr(cin,cin)
        feature_in = 64
        self.feat = nn.Sequential(
            nn.Conv2d(cin, feature_in*2, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(feature_in*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in * 2, feature_in, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(feature_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in, feature_in, kernel_size=3, bias=False, padding=1, stride=2),
            nn.BatchNorm2d(feature_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in, feature_in, kernel_size=3, bias=True, padding=1),
        )

        self.loc = nn.Sequential(
            nn.Linear(6400, 512),
            nn.Linear(512, 4),
        )

    def forward(self, xf,zf):
        feat1 = torch.cat([xf, zf],dim=1)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",feat1.size())
        feat = self.feat(feat1)

        b = feat.size()[0]
        feat = feat.view(b, -1)
        loc = self.loc(feat)
        return loc

class MaskPredHead(nn.Module):
    def __init__(self,chan_inbb,chan_incorr,hidden):
        super(MaskPredHead, self).__init__()
        self.fusion = nn.Conv2d(chan_incorr, chan_inbb, kernel_size=1, bias=True)
        self.head1 = nn.Sequential(
            nn.Conv2d(chan_inbb, hidden, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, bias=True, padding=1),
        )


    def forward(self, featbb,featcorr):
        fusion = self.fusion(featcorr)
        h,w = featbb.size()[2:]
        fusion = F.interpolate(fusion, (h,w), mode='bilinear', align_corners=False)
        featbb = featbb * F.sigmoid(fusion)
        mask = self.head1(featbb)
        return mask

class ModelBuilder(nn.Module):
    def __init__(self,cfg=None):
        super(ModelBuilder, self).__init__()

        # build backbone
        if cfg.BACKBONE.TYPE=="mobileones16outtwo":
            self.backbone = mobileones16outtwo(variant='s0')
        else:
            self.backbone = mobileone(variant='s0')
        # build adjust layer
        self.neck = AdjustAllLayerFPNAddFlagCenterCrop(in_channels=[128,256,1024],out_channels=[192,192,192])


        # build rpn head
        self.head = RPNHead(cin=384)
        # p = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)
        self.rank_cls_loss = rank_cls_loss()
        self.rank_loc_loss = rank_loc_loss()

    def template(self, z):
        temp_z = z.clone()
        temp_z -= 114.0
        temp_z /= 58.0
        zf = self.backbone(temp_z)
        zf = self.neck(zf, flagcentercrop=True)
        self.zf = zf


    def track(self, x):
        # import numpy as np
        # x = np.load("/home/ethan/SGS_IPU_SDK_v1.1.6/x_crop.npy").transpose((0, 3, 1, 2))
        # x = torch.from_numpy(x).cuda()
        # print(x.size(),"xxxx")
        imgsize = x.size()[2]
        temp_x = x.clone()
        temp_x -= 114.0
        temp_x /= 58.0
        xf = self.backbone(temp_x)
        xf = self.neck(xf)
        loc = self.head(xf, self.zf)
        return loc

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def forward(self, data,rank=None):
        """ only used in training
        """
        Nbatch, Nperbatch = data['search'].size()[:2]
        if rank is None:
            template = data['template'].cuda().float()
            search = data['search'].cuda().float().flatten(0,1)
            label_loc = data['label_loc'].cuda().float().flatten(0,1)
        else:
            template = data['template'].float().to(rank, non_blocking=True)
            search = data['search'].float().to(rank, non_blocking=True).flatten(0,1)
            label_loc = data['label_loc'].float().to(rank, non_blocking=True).flatten(0,1)

        imgsize = search.size()[2]
        if cfg.DATASET.CATDOGHORSETK.NBATCH_SHAPREPREV > 1:
            index_sel = []
            nstep = Nbatch // cfg.DATASET.CATDOGHORSETK.NBATCH_SHAPREPREV
            for i in range(nstep):
                idx = i * nstep + random.randint(0, cfg.DATASET.CATDOGHORSETK.NBATCH_SHAPREPREV - 1)
                index_sel.append(idx)
            if rank is None:
                index_sel = torch.tensor(index_sel).cuda()
            else:
                index_sel = torch.tensor(index_sel).to(rank)

            template = torch.index_select(template, 0, index_sel)

        Nimg = search.size()[0]
        template -= 114.0
        template /= 58.0
        search -= 114.0
        search /= 58.0
        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        zf = self.neck(zf, flagcentercrop=True)
        zf = torch.unsqueeze(zf, dim=1)
        zf = torch.tile(zf, [1, Nperbatch*cfg.DATASET.CATDOGHORSETK.NBATCH_SHAPREPREV, 1, 1, 1])
        zf = zf.flatten(0, 1)
        xf = self.neck(xf)
        loc = self.head(xf, zf)
        loc_loss = (loc - label_loc).abs().sum()
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['loc_loss'] = loc_loss.detach()
        return outputs

if __name__=='__main__':
    from siamban.models.backbone.mobileone_strideS16OutTwo import reparameterize_model_allskipscale
    from siamban.core.config import cfg

    cfg.BACKBONE.TYPE = "mobileones16outtwo"
    model = ModelBuilder()
    model.backbone = reparameterize_model_allskipscale(model.backbone)
    model = model.cuda()
    data = {}

    data['template'] = torch.randn((1,3,160,160)).cuda()
    data['search'] = torch.randn((1,5,3,160,160)).cuda()
    data['label_loc'] = torch.randn((1,5,4)).cuda()


    out= model(data)