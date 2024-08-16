# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

#from siamban.core.config import cfg
from siamban.models.rank_loss import select_cross_entropy_loss, select_iou_loss, rank_cls_loss,rank_loc_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck
from siamban.utils.point import Point
from siamban.models.backbone.mobileone_stride import mobileone
from siamban.models.backbone.mobileone_strideS16OutTwo import mobileone as mobileones16outtwo
from siamban.models.backbone.mobileone_strideS8S16OutTwo import mobileone as mobileones8s16outtwo

class AdjustAllLayerFlagCenterCrop(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=9):
        super(AdjustAllLayerFlagCenterCrop, self).__init__()
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

        out = downsample1 * weight[0] + downsample2 * weight[1]
        if flagcentercrop:
            l = (downsample1.size(3) - self.center_size) // 2
            r = l + self.center_size
            out = out[:, :, l:r, l:r]

        return out


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
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
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=3, bias=True, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, kernel_size=3, bias=True, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, kernel_size=3, bias=True, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)
                )

        self.xorr_search = nn.Conv2d(hidden, hidden, kernel_size=5, bias=False)
        self.xorr_kernel = nn.Conv2d(hidden, hidden, kernel_size=5, bias=True)

        self.xorr_activate = nn.Sequential(
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )

    def forward(self, kernel,search):
        kernel = self.conv_kernel(kernel)
        kernel_part = self.xorr_kernel(kernel)
        search = self.conv_search(search)
        search_part = self.xorr_search(search)
        feature = self.xorr_activate(search_part + kernel_part)
        out = self.head(feature)
        return out


class DepthwiseBAN(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, cls_out_channels=2):
        super(DepthwiseBAN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, cls_out_channels)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4)
    def forward(self, z_f,x_f):
        cls = self.cls(z_f,x_f)
        loc = self.loc(z_f,x_f)
        loc = torch.exp(loc)
        return cls, loc

class ModelBuilder(nn.Module):
    def __init__(self,cfg):
        super(ModelBuilder, self).__init__()

        self.cfg = cfg
        if cfg.BACKBONE.TYPE=="mobileones16outtwo":
            self.backbone = mobileones16outtwo(variant='s0')
        elif cfg.BACKBONE.TYPE=='mobileones8s16outtwo':
            self.backbone = mobileones8s16outtwo(variant='s0')
        else:
            self.backbone = mobileone(variant='s0')
        channel = cfg.BAN.channels
        # build adjust layer
        self.neck = AdjustAllLayerFlagCenterCrop(in_channels=[256,1024],out_channels=[channel,channel],center_size=7)

        # build ban head
        self.head = DepthwiseBAN(in_channels=channel, out_channels=channel)
        self.rank_cls_loss=rank_cls_loss()
        self.rank_loc_loss=rank_loc_loss()
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)

    def template(self, z):
        # self.bbox = bbox
        # print(z.min().data.cpu().numpy(),z.max().data.cpu().numpy(),z.mean().data.cpu().numpy())
        if self.cfg.TRAIN.DATA_NORMALIZE_MODE == 0:
            z /= 255.0
            z -= self.mean
            z /= self.std
        elif self.cfg.TRAIN.DATA_NORMALIZE_MODE == 1:
            z -= 114.0
            z /= 58.0
        zf = self.backbone(z)
        zf = self.neck(zf[1:],flagcentercrop=True)
        self.zf = zf
        # print("zf:",self.zf.min().data.cpu().numpy(),self.zf.max().data.cpu().numpy(),self.zf.mean().data.cpu().numpy(),self.zf.size())

    def track(self, x):
        # print("x:",x.min().data.cpu().numpy(),x.max().data.cpu().numpy(),x.mean().data.cpu().numpy())
        if self.cfg.TRAIN.DATA_NORMALIZE_MODE == 0:
            x /= 255.0
            x -= self.mean
            x /= self.std
        elif self.cfg.TRAIN.DATA_NORMALIZE_MODE == 1:
            x -= 114.0
            x /= 58.0
        xf = self.backbone(x)
        xf = self.neck(xf[1:])
        # print("xfafterneck:",xf.min().data.cpu().numpy(),xf.max().data.cpu().numpy(),xf.mean().data.cpu().numpy())
        # print("self.zf:",self.zf.min().data.cpu().numpy(),self.zf.max().data.cpu().numpy(),self.zf.mean().data.cpu().numpy())

        cls, loc = self.head(self.zf,xf)
        # cls = cls.sigmoid()
        # print("cls:",cls.min().data.cpu().numpy(),cls.max().data.cpu().numpy(),cls.mean().data.cpu().numpy(),cls.size())
        # print("loc:",loc.min().data.cpu().numpy(),loc.max().data.cpu().numpy(),loc.mean().data.cpu().numpy(),loc.size())
        # exit()

        return cls,loc

    def log_softmax(self, cls):
        if self.cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            # cls = F.log_softmax(cls, dim=3)
            if self.cfg.TRAIN.FLAG_SIGMOID_LOSS:
                # cls = F.logsigmoid(cls)
                cls = F.sigmoid(cls)
            else:
                cls = F.log_softmax(cls, dim=3)
        return cls

    def convert_bbox(self,delta, points):
        batch_size=delta.shape[0]
        delta = delta.view(batch_size, 4, -1) #delta shape before: [batch_size,4,25,25]
        points=points.view(2,-1) #points shape before: [2,25,25]
        output_boxes=torch.zeros(batch_size,4,delta.shape[2])
        for i in range (batch_size):
           output_boxes[i][0, :] = points[0,:] - delta[i][0, :]
           output_boxes[i][1, :] = points[1,:] - delta[i][1, :]
           output_boxes[i][2, :] = points[0,:] + delta[i][2, :]
           output_boxes[i][3, :] = points[1,:] + delta[i][3, :]
        return output_boxes

    def forward(self, data,rank=None):
        """ only used in training
        """
        if rank is None:
            template = data['template'].cuda()
            search = data['search'].cuda()
            label_cls = data['label_cls'].cuda()
            label_loc = data['label_loc'].cuda()
            label_target = data['search_bbox'].cuda()
        else:
            template = data['template'].to(rank)
            search = data['search'].to(rank)
            label_cls = data['label_cls'].to(rank)
            label_loc = data['label_loc'].to(rank)
            label_target = data['search_bbox'].to(rank)

        # template_box = data['template_box'].cuda()
        # init_box = self.cornercenter(template_box)

        if self.cfg.TRAIN.DATA_NORMALIZE_MODE == 0:
            template /= 255.0
            search /= 255.0
            template -= self.mean
            template /= self.std
            search -= self.mean
            search /= self.std
        elif self.cfg.TRAIN.DATA_NORMALIZE_MODE == 1:
            template -= 114.0
            template /= 58.0
            search -= 114.0
            search /= 58.0

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        zf = self.neck(zf[1:],flagcentercrop=True)
        xf = self.neck(xf[1:])
        cls, loc = self.head(zf,xf)



        point_tensor = torch.from_numpy(self.points.points).cuda().view(1, 2, -1)
        Nbatch = loc.size()[0]
        delta = loc.view(Nbatch, 4, -1)  # delta shape before: [batch_size,4,25,25]
        pred_bboxes = delta.clone()
        pred_bboxes[:, :2] = point_tensor - delta[:, :2]
        pred_bboxes[:, 2:] = point_tensor + delta[:, 2:]

        # cls loss with cross entropy loss
        cls = self.log_softmax(cls)
        # cls_loss = select_cross_entropy_loss(cls, label_cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls, self.cfg.TRAIN.FLAG_SIGMOID_LOSS, rank=rank)

        # loc loss with iou loss
        loc_loss = select_iou_loss(loc, label_loc, label_cls, rank=rank)

        outputs = {}
        outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                self.cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if self.cfg.TRAIN.RANK_CLS_WEIGHT > 0:
            CR_loss = self.rank_cls_loss(cls, label_cls, rank=rank, flagsigmoid=self.cfg.TRAIN.FLAG_SIGMOID_LOSS)
            outputs['total_loss'] = outputs['total_loss'] + self.cfg.TRAIN.RANK_CLS_WEIGHT * CR_loss
            outputs['CR_loss'] = self.cfg.TRAIN.RANK_CLS_WEIGHT * CR_loss

        if self.cfg.TRAIN.RANK_IGR_WEIGHT > 0:
            IGR_loss_1, IGR_loss_2 = self.rank_loc_loss(cls, label_cls, pred_bboxes, label_target, rank=rank,
                                                        flagsigmoid=self.cfg.TRAIN.FLAG_SIGMOID_LOSS)
            outputs['total_loss'] = outputs[
                                        'total_loss'] + self.cfg.TRAIN.RANK_IGR_WEIGHT * IGR_loss_1 + self.cfg.TRAIN.RANK_IGR_WEIGHT * IGR_loss_2

            outputs['IGR_loss_1'] = self.cfg.TRAIN.RANK_IGR_WEIGHT * IGR_loss_1
            outputs['IGR_loss_2'] = self.cfg.TRAIN.RANK_IGR_WEIGHT * IGR_loss_2
        #print(outputs['total_loss'])

        return outputs
