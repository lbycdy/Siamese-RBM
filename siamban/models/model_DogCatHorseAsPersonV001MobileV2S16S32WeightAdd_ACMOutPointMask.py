# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.rank_loss import select_cross_entropy_loss, select_iou_loss, rank_cls_loss,rank_loc_loss

from siamban.models.backbone.mobile_v2 import mobilenetv2

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
    def __init__(self, in_channels, out_channels, center_size=9):
        super(AdjustAllLayerFPNAddFlagCenterCrop, self).__init__()
        self.center_size = center_size
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels[1], 1, bias=False),
            nn.BatchNorm2d(out_channels[1]),
        )
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, features,flagcentercrop=False):
        downsample1 = self.downsample1(features[0])
        downsample2 = self.downsample2(features[1])
        weight = F.softmax(self.weight, 0)

        out = downsample1*weight[0]+downsample2*weight[1]
        if flagcentercrop:
            l = (out.size(3) - self.center_size) // 2
            r = l + self.center_size
            out = out[:, :, l:r, l:r]
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

    def forward(self, search,kernel):
        kernel = self.conv_kernel(kernel)
        kernel_part = self.xorr_kernel(kernel)
        search = self.conv_search(search)
        search_part = self.xorr_search(search)
        tmp = search_part*kernel_part
        feature = self.xorr_activate(tmp)
        return feature

class RPNHead(nn.Module):
    def __init__(self,cin = 256):
        super(RPNHead, self).__init__()
        self.depthcorr = DepthwiseXCorr(cin,cin)
        feature_in = 32
        self.head1 = nn.Sequential(
            nn.Conv2d(cin, feature_in*2, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(feature_in*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in * 2, feature_in, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(feature_in),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(feature_in, feature_in,kernel_size=3, bias=False,padding=1),
            nn.BatchNorm2d(feature_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in, 2, kernel_size=3, bias=True, padding=1),
        )
        self.loc = nn.Sequential(
            nn.Conv2d(feature_in, feature_in, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(feature_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in, 4, kernel_size=3, bias=True, padding=1),

        )



    def forward(self, xf,zf):
        depthcorr = self.depthcorr(xf,zf)
        head1 = self.head1(depthcorr)
        cls = self.cls(head1)
        loc = self.loc(head1)
        return cls,torch.exp(loc)

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
        self.backbone = mobilenetv2(used_layers=[5, 7])
        # build adjust layer
        self.neck = AdjustAllLayerFPNAddFlagCenterCrop(in_channels=[96,320],out_channels=[192,192],center_size=9)


        # build rpn head
        self.head = RPNHead(cin=192)
        # p = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)
        self.rank_cls_loss = rank_cls_loss()
        self.rank_loc_loss = rank_loc_loss()

    def template(self, z):
        z -= 114.0
        z /= 58.0
        zf = self.backbone(z)
        zf = self.neck(zf[1:], flagcentercrop=True)
        self.zf = zf


    def track(self, x):
        # import numpy as np
        # x = np.load("/home/ethan/SGS_IPU_SDK_v1.1.6/x_crop.npy").transpose((0, 3, 1, 2))
        # x = torch.from_numpy(x).cuda()
        # print(x.size(),"xxxx")
        x -= 114.0
        x /= 58.0
        xf = self.backbone(x)
        xf = self.neck(xf[1:])
        cls, loc = self.head(xf, self.zf)

        return cls,loc

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        cls = cls.permute(0, 2, 3, 1).contiguous()
        if cfg.TRAIN.FLAG_SIGMOID_LOSS:
            cls = F.sigmoid(cls)
        else:
            cls = F.log_softmax(cls, dim=3)
        return cls

    # def convert_bbox(self, delta, points):
    #     batch_size = delta.shape[0]
    #     delta = delta.view(batch_size, 4, -1)  # delta shape before: [batch_size,4,25,25]
    #     points = points.view(1,2, -1)  # points shape before: [2,25,25]
    #     output_boxes = torch.zeros(batch_size, 4, delta.shape[2])
    #     print(output_boxes.size(),points.size(),delta.size())
    #     #torch.Size([175, 4, 121]) torch.Size([2, 121]) torch.Size([175, 4, 121])
    #     # for i in range(batch_size):
    #     #     output_boxes[i][0, :] = points[0, :] - delta[i][0, :]
    #     #     output_boxes[i][1, :] = points[1, :] - delta[i][1, :]
    #     #     output_boxes[i][2, :] = points[0, :] + delta[i][2, :]
    #     #     output_boxes[i][3, :] = points[1, :] + delta[i][3, :]
    #     output_boxes[:,:2] = points - delta[:,:2]
    #     output_boxes[:,2:] = points  + delta[:,2:]
    #
    #     return output_boxes

    def forward(self, data,rank=None):
        """ only used in training
        """
        if rank is None:
            template = data['template'].cuda().float().flatten(0,1)
            search = data['search'].cuda().float().flatten(0,1)
            label_loc = data['label_loc'].cuda().float().flatten(0,1)
            label_cls = data['label_cls'].cuda().flatten(0,1)
            label_target = data['searchboxes'].cuda().float().flatten(0,1)
        else:
            template = data['template'].float().to(rank, non_blocking=True).flatten(0,1)
            search = data['search'].float().to(rank, non_blocking=True).flatten(0,1)
            label_loc = data['label_loc'].float().to(rank, non_blocking=True).flatten(0,1)
            label_cls = data['label_cls'].to(rank, non_blocking=True).flatten(0,1)
            label_target = data['searchboxes'].to(rank, non_blocking=True).float().flatten(0,1)
        # print(template.size(),search.size(),label_cls.size(),label_loc.size(),label_target.size())
        # print(torch.unique(label_cls))
        Nbatch = template.size()[0]
        template -= 114.0
        template /= 58.0
        search -= 114.0
        search /= 58.0
        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        zf = self.neck(zf, flagcentercrop=True)
        xf = self.neck(xf)
        cls,loc = self.head(xf,zf)
        point_tensor = torch.from_numpy(self.points.points).to(rank).view(1, 2, -1)
        delta = loc.view(Nbatch, 4, -1)  # delta shape before: [batch_size,4,25,25]
        pred_bboxes = delta.clone()
        pred_bboxes[:, :2] = point_tensor - delta[:, :2]
        pred_bboxes[:, 2:] = point_tensor + delta[:, 2:]

        cls = self.log_softmax(cls)

        cls_loss = select_cross_entropy_loss(cls, label_cls, cfg.TRAIN.FLAG_SIGMOID_LOSS,rank)
        loc_loss = select_iou_loss(loc, label_loc, label_cls,rank=rank)
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss.detach()
        outputs['loc_loss'] = loc_loss.detach()
        if cfg.TRAIN.RANK_CLS_WEIGHT>0:
            CR_loss = self.rank_cls_loss(cls, label_cls,flagsigmoid=cfg.TRAIN.FLAG_SIGMOID_LOSS,rank=rank)

            # try:
            outputs['total_loss'] += cfg.TRAIN.RANK_CLS_WEIGHT * CR_loss
            outputs['CR_loss'] = CR_loss.detach()
            # except:
            #     pass
        if cfg.TRAIN.RANK_IGR_WEIGHT>0:
            IGR_loss_1, IGR_loss_2 = self.rank_loc_loss(cls, label_cls, pred_bboxes, label_target,
                                                    flagsigmoid=cfg.TRAIN.FLAG_SIGMOID_LOSS, rank=rank)
            # try:
            outputs['total_loss'] += (cfg.TRAIN.RANK_IGR_WEIGHT * IGR_loss_1 + cfg.TRAIN.RANK_IGR_WEIGHT * IGR_loss_2)
            outputs['IGR_loss_1'] = IGR_loss_1.detach()
            outputs['IGR_loss_2'] = IGR_loss_2.detach()
            # except:
            #     pass



        return outputs

if __name__=='__main__':
    cfg.BACKBONE.TYPE = 'darknet20190121conv5mask'
    backbone = mobilenetv2(used_layers=[5, 7])
    t = torch.randn(5,3,160,160)
    out = backbone(t)
    for o in out:
        print(o.size())
    # model = ModelBuilder().cuda()
    # data = {}
    # data['template'] = torch.randn((1,5,3,160,160))
    # data['search'] = torch.randn((1,5,3,160,160))
    # data['label_loc'] = torch.randn((1,5,4))
    # data['label_cls'] = torch.randn((1,5,1))

    # out= model(data)