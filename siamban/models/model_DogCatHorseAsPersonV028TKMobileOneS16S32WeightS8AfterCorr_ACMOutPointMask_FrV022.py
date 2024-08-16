# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.rank_loss import select_cross_entropy_loss, select_iou_loss, rank_cls_loss,rank_loc_loss,rank_loc_loss_fast,rank_center_loc_loss

from siamban.models.backbone.mobileone_strideS16OutTwo import mobileone as mobileones16outtwo
from siamban.models.backbone.mobileone_strideS16OutTwoS8OutMid import mobileone as mobileones16outtwos8outmid

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
    def __init__(self,cin = 256,cinf8=128,p=0.0,pspatial=0.0,flagclassoncorr=False):
        super(RPNHead, self).__init__()
        self.depthcorr = DepthwiseXCorr(cin,cin)
        feature_in = 32
        self.head1 = nn.Sequential(
            nn.Conv2d(cin+cinf8, feature_in*2, kernel_size=1, bias=False, padding=0),
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
        self.flagclassoncorr = flagclassoncorr
        if self.flagclassoncorr:
            self.clsoncorr = nn.Conv2d(cin, 2, kernel_size=3, bias=True, padding=1)


        self.loc_scale = nn.Parameter(torch.ones(1))
        self.drop = nn.Dropout2d(p=p)
        self.dropspatial = nn.Dropout2d(p=pspatial)

    def forward(self, xf,zf,f8):
        depthcorr = self.depthcorr(xf,zf)
        if self.flagclassoncorr:
            clsoncorr = self.clsoncorr(depthcorr)
        else:
            clsoncorr = None
        depthcorr = self.drop(depthcorr)
        featcat = torch.cat([depthcorr,f8],dim=1)
        head1 = self.head1(featcat)

        n, c, h, w = head1.size()
        head1drop = head1.view(n, c, h * w).transpose(1, 2).contiguous().view(n, h * w, c, 1)
        head1drop = self.dropspatial(head1drop).transpose(1, 2).view(n, c, h, w)
        cls = self.cls(head1drop)
        loc = self.loc(head1drop)
        return cls,torch.exp(loc*self.loc_scale[0]),clsoncorr

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

        self.cfg = cfg
        # build backbone
        if cfg.BACKBONE.TYPE=="mobileones16outtwo":
            self.backbone = mobileones16outtwo(variant='s0')
        elif cfg.BACKBONE.TYPE=='mobileones16outtwos8outmid':
            self.backbone = mobileones16outtwos8outmid(variant='s0')
        # build adjust layer
        self.neck = AdjustAllLayerFPNAddFlagCenterCrop(in_channels=[256,1024],out_channels=[192,192],center_size=9)


        # build rpn head
        self.head = RPNHead(cin=192,p=cfg.TRAIN.PROB_DROPOUT,pspatial=cfg.TRAIN.PROB_DROPOUTSPATIAL,flagclassoncorr=cfg.TRAIN.FLAGCLASSONCORR)
        # p = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)
        self.rank_cls_loss = rank_cls_loss()
        self.rank_loc_loss = rank_loc_loss()
        self.rank_loc_loss_fast = rank_loc_loss_fast(cfg)
        self.rank_loc_loss_center = rank_center_loc_loss(cfg)


    def template(self, z):
        netin = z.clone()
        netin -= 114.0
        netin /= 58.0
        print("netinz:",netin.min(),netin.max(),netin.mean())
        zf = self.backbone(netin)
        zf = self.neck(zf[1:], flagcentercrop=True)
        self.zf = zf
        print("zf:",zf.min(),zf.max(),zf.mean())



    def track(self, x):
        # import numpy as np
        # x = np.load("/home/ethan/SGS_IPU_SDK_v1.1.6/x_crop.npy").transpose((0, 3, 1, 2))
        # x = torch.from_numpy(x).cuda()
        # print(x.size(),"xxxx")
        imgsize = x.size()[2]
        netin = x.clone()
        netin -= 114.0
        netin /= 58.0
        print("netinx:",netin.min(),netin.max(),netin.mean())

        xfbb = self.backbone(netin)
        xf = self.neck(xfbb[1:])
        cls, loc,clsoncorr = self.head(xf, self.zf,xfbb[0])
        if self.cfg.TRAIN.NORM_REG_TARGETS:
            loc *= (float(imgsize)/10.0)
        # loc *= 16
        return cls,loc,clsoncorr

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
        point_tensor = torch.from_numpy(self.points.points).to(rank).view(1, 2, -1)

        if self.cfg.TRAIN.NORM_REG_TARGETS:
            label_target /= (float(imgsize)/10.0)
            label_loc /= (float(imgsize)/10.0)
            point_tensor /= (float(imgsize)/10.0)
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
        xfbb = self.backbone(search)
        zf = self.neck(zf[1:], flagcentercrop=True)
        zf = torch.unsqueeze(zf, dim=1)
        zf = torch.tile(zf, [1, Nperbatch*cfg.DATASET.CATDOGHORSETK.NBATCH_SHAPREPREV, 1, 1, 1])
        zf = zf.flatten(0, 1)
        xf = self.neck(xfbb[1:])
        cls, loc,clsoncorr = self.head(xf,zf,xfbb[0])
        # print(cls.size(),loc.size())
        delta = loc.view(Nimg, 4, -1)  # delta shape before: [batch_size,4,25,25]
        pred_bboxes = delta.clone()

        pred_bboxes[:, :2] = point_tensor - delta[:, :2]
        pred_bboxes[:, 2:] = point_tensor + delta[:, 2:]

        cls = self.log_softmax(cls)

        cls_loss = select_cross_entropy_loss(cls, label_cls, cfg.TRAIN.FLAG_SIGMOID_LOSS,rank)
        loc_loss = select_iou_loss(loc, label_loc, label_cls, rank=rank)
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss.detach()
        outputs['loc_loss'] = loc_loss.detach()
        if self.cfg.TRAIN.FLAGCLASSONCORR:
            clsoncorr = self.log_softmax(clsoncorr)
            clsoncorr_loss = select_cross_entropy_loss(clsoncorr, label_cls, cfg.TRAIN.FLAG_SIGMOID_LOSS, rank)
            outputs['total_loss'] += clsoncorr_loss
            outputs['clsoncorr_loss'] = clsoncorr_loss.detach()

        """
        File "/home/inspur/anaconda3/envs/pt1/lib/python3.6/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
        File "/home/inspur/anaconda3/envs/pt1/lib/python3.6/site-packages/torch/nn/parallel/distributed.py", line 886, in forward
        output = self.module(*inputs[0], **kwargs[0])
        File "/home/inspur/anaconda3/envs/pt1/lib/python3.6/site-packages/torch/nn/modules/module.py", line 1102, in _call_impl
        return forward_call(*input, **kwargs)
        File "/home/inspur/work/siamban-acm/siamban/models/model_DogCatHorseAsPersonV000MobileOneS16S32WeightAdd_ACMOutPointMask.py", line 274, in forward
        outputs['total_loss'] += cfg.TRAIN.RANK_CLS_WEIGHT * CR_loss
        RuntimeError: output with shape [] doesn't match the broadcast shape [1]
        """
        if cfg.TRAIN.RANK_CLS_WEIGHT>0:
            CR_loss = self.rank_cls_loss(cls, label_cls,flagsigmoid=cfg.TRAIN.FLAG_SIGMOID_LOSS,rank=rank)

            # try:
            outputs['total_loss'] += cfg.TRAIN.RANK_CLS_WEIGHT * CR_loss
            outputs['CR_loss'] = CR_loss.detach()
            # except:
            #     pass
        if cfg.TRAIN.RANK_IGR_WEIGHT>0:
            if cfg.TRAIN.RANK_IGR_MODE==0:
                IGR_loss_1, IGR_loss_2 = self.rank_loc_loss(cls, label_cls, pred_bboxes, label_target,
                                                    flagsigmoid=cfg.TRAIN.FLAG_SIGMOID_LOSS, rank=rank)
            elif cfg.TRAIN.RANK_IGR_MODE==1:
                IGR_loss_1, IGR_loss_2 = self.rank_loc_loss_fast(cls, label_cls, pred_bboxes, label_target,
                                                            flagsigmoid=cfg.TRAIN.FLAG_SIGMOID_LOSS, rank=rank)
            elif cfg.TRAIN.RANK_IGR_MODE==2:
                IGR_loss_1, IGR_loss_2 = self.rank_loc_loss_center(cls, label_cls, label_loc, pred_bboxes, label_target,
                                                            flagsigmoid=cfg.TRAIN.FLAG_SIGMOID_LOSS, rank=rank)

            # try:
            outputs['total_loss'] += (cfg.TRAIN.RANK_IGR_WEIGHT * IGR_loss_1 + cfg.TRAIN.RANK_IGR_WEIGHT * IGR_loss_2)
            outputs['IGR_loss_1'] = IGR_loss_1.detach()
            outputs['IGR_loss_2'] = IGR_loss_2.detach()
            # except:
            #     pass



        return outputs

if __name__=='__main__':
    from siamban.models.backbone.mobileone_strideS16OutTwo import reparameterize_model_allskipscale
    from siamban.core.config import cfg

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