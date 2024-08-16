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

class AdjustFlagCenterCrop(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=9):
        super(AdjustFlagCenterCrop, self).__init__()
        self.center_size = center_size
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, features,flagcentercrop=False):
        downsample1 = self.downsample1(features)
        if flagcentercrop:
            l = (downsample1.size(3) - self.center_size) // 2
            r = l + self.center_size
            downsample1 = downsample1[:, :, l:r, l:r]

        return downsample1



class MultiDilation(nn.Module):
    def __init__(self, in_channels, hidden):
        super(MultiDilation, self).__init__()
        self.xf11 = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.xf22 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False,dilation=2),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.zf11 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.zf22 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, bias=False,dilation=2),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, z,x):
        zf11 = self.zf11(z)
        zf22 = self.zf22(z)
        xf11 = self.xf11(x)
        xf22 = self.xf22(x)

        return [zf11,zf22],[xf11,xf22]

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, kernel_size=3,kernacm=5):
        super(DepthwiseXCorr, self).__init__()
        self.headcls = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=3, bias=True, padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 2, kernel_size=3, padding=1)
                )
        self.headloc = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 4, kernel_size=3, padding=1)
        )

        self.xorr_search = nn.Conv2d(in_channels, hidden, kernel_size=kernacm, bias=False)
        self.xorr_kernel = nn.Conv2d(in_channels, hidden, kernel_size=kernacm, bias=True)

        self.xorr_activate = nn.Sequential(
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )

    def forward(self, kernel,search):
        kernel_part = self.xorr_kernel(kernel)
        search_part = self.xorr_search(search)
        feature = self.xorr_activate(search_part + kernel_part)
        cls = self.headcls(feature)
        loc = self.headloc(feature)

        return cls,loc


class MutltiBAN(nn.Module):
    def __init__(self, in_channels, hidden):
        super(MutltiBAN, self).__init__()
        self.multidilation = MultiDilation(in_channels, hidden)
        self.box1 = DepthwiseXCorr(in_channels, hidden, kernel_size=3,kernacm=5)
        self.box2 = DepthwiseXCorr(in_channels, hidden, kernel_size=3,kernacm=3)

        self.cls_weight = nn.Parameter(torch.ones(2))
        self.loc_weight = nn.Parameter(torch.ones(2))
        self.loc_scale = nn.Parameter(torch.ones(2))

    def forward(self, zf,xf):
        z_fs,x_fs = self.multidilation(zf,xf)
        # print(z_fs[0].size(),z_fs[1].size(),x_fs[0].size(),x_fs[1].size())
        cls = 0
        loc = 0
        cls_weight = F.softmax(self.cls_weight, 0)
        loc_weight = F.softmax(self.loc_weight, 0)
        c,l = self.box1(z_fs[0],x_fs[0])
        cls += cls_weight[0] * c
        loc += loc_weight[0] * l * self.loc_scale[0]
        c, l = self.box2(z_fs[1], x_fs[1])
        cls += cls_weight[1] * c
        loc += loc_weight[1] * l * self.loc_scale[1]

        loc = torch.exp(loc)
        return cls,loc

class ModelBuilder(nn.Module):
    def __init__(self,cfg):
        super(ModelBuilder, self).__init__()

        self.cfg = cfg
        self.backbone = mobileone(variant='s0')
        channel = cfg.BAN.channels
        # build adjust layer
        self.neck = AdjustFlagCenterCrop(in_channels=256,out_channels=channel,center_size=7)

        # build ban head
        self.head = MutltiBAN(channel, channel)
        self.rank_cls_loss=rank_cls_loss()
        self.rank_loc_loss=rank_loc_loss()
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE // 2)
        self._init_weights_gauss()

    def template(self, z):
        # self.bbox = bbox
        # print(z.min().data.cpu().numpy(),z.max().data.cpu().numpy(),z.mean().data.cpu().numpy())
        zf = self.backbone(z)
        zf = self.neck(zf,flagcentercrop=True)
        self.zf = zf
        # print("zf:",self.zf.min().data.cpu().numpy(),self.zf.max().data.cpu().numpy(),self.zf.mean().data.cpu().numpy(),self.zf.size())

    def track(self, x):
        # print("x:",x.min().data.cpu().numpy(),x.max().data.cpu().numpy(),x.mean().data.cpu().numpy())

        xf = self.backbone(x)
        xf = self.neck(xf)
        # print("xfafterneck:",xf.min().data.cpu().numpy(),xf.max().data.cpu().numpy(),xf.mean().data.cpu().numpy())
        # print("self.zf:",self.zf.min().data.cpu().numpy(),self.zf.max().data.cpu().numpy(),self.zf.mean().data.cpu().numpy())

        cls, loc = self.head(self.zf,xf)
        # cls = cls.sigmoid()
        # print("cls:",cls.min().data.cpu().numpy(),cls.max().data.cpu().numpy(),cls.mean().data.cpu().numpy(),cls.size())
        # print("loc:",loc.min().data.cpu().numpy(),loc.max().data.cpu().numpy(),loc.mean().data.cpu().numpy(),loc.size())
        # exit()

        return {
                'cls': cls,
                'loc': loc
               }

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

        zf = self.neck(zf[1],flagcentercrop=True)
        xf = self.neck(xf[1])
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

        CR_loss=self.rank_cls_loss(cls,label_cls, rank=rank,flagsigmoid=self.cfg.TRAIN.FLAG_SIGMOID_LOSS)
        IGR_loss_1,IGR_loss_2=self.rank_loc_loss(cls,label_cls,pred_bboxes,label_target, rank=rank,flagsigmoid=self.cfg.TRAIN.FLAG_SIGMOID_LOSS)
        outputs = {}
        outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            self.cfg.TRAIN.LOC_WEIGHT * loc_loss +self.cfg.TRAIN.RANK_CLS_WEIGHT*CR_loss+self.cfg.TRAIN.RANK_IGR_WEIGHT*IGR_loss_1+self.cfg.TRAIN.RANK_IGR_WEIGHT*IGR_loss_2
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['CR_loss'] = self.cfg.TRAIN.RANK_CLS_WEIGHT*CR_loss
        outputs['IGR_loss_1'] =self.cfg.TRAIN.RANK_IGR_WEIGHT*IGR_loss_1
        outputs['IGR_loss_2'] = self.cfg.TRAIN.RANK_IGR_WEIGHT*IGR_loss_2
        #print(outputs['total_loss'])

        return outputs

    def _init_weights_gauss(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()