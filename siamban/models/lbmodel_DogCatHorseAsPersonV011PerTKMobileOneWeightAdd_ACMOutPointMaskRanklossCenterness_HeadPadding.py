# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.rank_loss1 import select_cross_entropy_loss, select_iou_loss, rank_cls_loss,rank_loc_loss,select_cross_entropy_centerness_loss, select_iou_centerness_loss

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
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )

        self.xorr_search = nn.Conv2d(hidden, hidden, kernel_size=7, bias=False,padding=3)
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
            nn.Conv2d(feature_in, feature_in, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(feature_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in, 2, kernel_size=3, bias=True, padding=1),
        )
        self.loc = nn.Sequential(
            nn.Conv2d(feature_in, feature_in, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(feature_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in, feature_in, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(feature_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in, 4, kernel_size=3, bias=True, padding=1),

        )
        # self.centerness = nn.Sequential(
        #     nn.Conv2d(feature_in, feature_in, kernel_size=3, bias=False, padding=1),
        #     nn.BatchNorm2d(feature_in),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(feature_in, 1, kernel_size=3, bias=True, padding=1),
        # )

        self.loc_scale = nn.Parameter(torch.ones(1))

        self.drop = nn.Dropout2d(p=cfg.BACKBONE.DROPOUT)
    def forward(self, xf,zf):
        depthcorr = self.depthcorr(xf,zf)
        depthcorr = self.drop(depthcorr)
        head1 = self.head1(depthcorr)
        cls = self.cls(head1)
        loc = self.loc(head1)
        # centerness = self.centerness(head1)
        return cls,torch.exp(loc*self.loc_scale[0])

        # return cls, torch.exp(loc * self.loc_scale[0]), centerness


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
        self.neck = AdjustAllLayerFPNAddFlagCenterCrop(in_channels=[128,256,1024],out_channels=[192,192,192],center_size=9)


        # build rpn head
        self.head = RPNHead(cin=192)
        # p = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)
        self.rank_cls_loss = rank_cls_loss()
        self.rank_loc_loss = rank_loc_loss()
        self.centerness_loss = nn.BCEWithLogitsLoss()
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
        cls, loc = self.head(xf, self.zf)
        # loc *= (float(imgsize)/10.0)
        loc *= 16
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

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def select_centerness_loss(self, label_loc, label_cls):
        batch_size = label_cls.shape[0]

        label_cls = label_cls.reshape(-1)
        pos = label_cls.data.eq(1).nonzero().squeeze().cuda()
        label_loc = label_loc.view(batch_size, 4, -1)

        # label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
        label_loc = torch.index_select(label_loc, 0, pos)
        centerness_targets = self.compute_centerness_targets(label_loc)
        return centerness_targets



    def forward(self, data,rank=None):
        """ only used in training
        """
        Nbatch, Nperbatch = data['search'].size()[:2]
        if rank is None:
            template = data['template'].cuda().float()
            search = data['search'].cuda().float().flatten(0,1)
            label_loc = data['label_loc'].cuda().float().flatten(0,1)
            label_cls = data['label_cls'].cuda().flatten(0,1)
            label_target = data['searchboxes'].cuda().float().flatten(0,1)
        else:
            template = data['template'].float().to(rank, non_blocking=True)
            search = data['search'].float().to(rank, non_blocking=True).flatten(0,1)
            label_loc = data['label_loc'].float().to(rank, non_blocking=True).flatten(0,1)
            label_cls = data['label_cls'].to(rank, non_blocking=True).flatten(0,1)
            label_target = data['searchboxes'].to(rank, non_blocking=True).float().flatten(0,1)
        # print(template.size(),search.size(),label_cls.size(),label_loc.size(),label_target.size())
        # print(torch.unique(label_cls))
        imgsize = search.size()[2]
        label_target /= (float(imgsize)/10.0)
        label_loc /= (float(imgsize)/10.0)
        point_tensor = torch.from_numpy(self.points.points).to(rank).view(1, 2, -1)/(float(imgsize)/10.0)
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
        cls, loc = self.head(xf,zf)
        # print(cls.size(),loc.size())
        delta = loc.view(Nimg, 4, -1)  # delta shape before: [batch_size,4,25,25]
        pred_bboxes = delta.clone()
        point_tensor = point_tensor.cuda()
        pred_bboxes[:, :2] = point_tensor - delta[:, :2]
        pred_bboxes[:, 2:] = point_tensor + delta[:, 2:]

        cls = self.log_softmax(cls)

        centerness_target = self.select_centerness_loss(label_loc, label_cls)
        cls_loss = select_cross_entropy_centerness_loss(cls, label_cls, centerness_target, cfg.TRAIN.FLAG_SIGMOID_LOSS,rank)
        loc_loss = select_iou_centerness_loss(loc, label_loc, centerness_target, label_cls, rank=rank)


        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss.detach()
        outputs['loc_loss'] = loc_loss.detach()


        return outputs

if __name__=='__main__':
    from siamban.models.backbone.mobileone_strideS16OutTwo import reparameterize_model_allskipscale
    from siamban.core.config import cfg

    cfg.TRAIN.OUTPUT_SIZE = 19
    cfg.BACKBONE.TYPE = "mobileones16outtwo"
    model = ModelBuilder(cfg)
    model.backbone = reparameterize_model_allskipscale(model.backbone)
    model = model.cuda()
    data = {}
    data['template'] = torch.randn((1,3,160,160)).cuda()
    data['search'] = torch.randn((1,5,3,160,160)).cuda()
    data['label_loc'] = torch.randn((1,5,4)).cuda()
    data['label_cls'] = torch.randn((1,5,1)).cuda()
    data['searchboxes'] = torch.randn((1,5,4)).cuda()

    out= model(data)
    # loc = torch.randn((1,4,19,19))
    # cls = torch.randn((1,2,19,19))