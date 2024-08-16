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
from siamban.models.backbone.mobileone_strideS16OutTwo import mobileone

from siamban.core.xcorr import xcorr_fast, xcorr_depthwise




#################################################################################################################3


class AdjustAllLayerFPNThreeInAddFlagCenterCrop(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=9):
        super(AdjustAllLayerFPNThreeInAddFlagCenterCrop, self).__init__()
        self.center_size = center_size
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels[1], 1, bias=False),
            nn.BatchNorm2d(out_channels[1]),
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, bias=False),
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
                nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1)
                )

        self.xorr_search = nn.Conv2d(hidden, hidden, kernel_size=5, bias=False)
        self.xorr_kernel = nn.Conv2d(hidden, hidden, kernel_size=5, bias=True)

        self.xorr_activate = nn.Sequential(
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



class ModelBuilder(nn.Module):
    def __init__(self,cfg):
        super(ModelBuilder, self).__init__()

        self.cfg = cfg
        self.backbone = mobileone(variant='s0')
        channel = cfg.BAN.channels
        # build adjust layer
        self.neck = AdjustAllLayerFPNThreeInAddFlagCenterCrop(in_channels=[128,256,1024],out_channels=[channel,channel,channel],center_size=7)

        # build ban head
        self.head = DepthwiseXCorr(in_channels=channel, hidden=channel, out_channels=256, kernel_size=3)


        self._init_weights_gauss()

    def template(self, z):
        z -= 114.0
        z /= 58.0
        # self.bbox = bbox
        # print(z.min().data.cpu().numpy(),z.max().data.cpu().numpy(),z.mean().data.cpu().numpy())
        zf = self.backbone(z)
        zf = self.neck(zf,flagcentercrop=True)
        self.zf = zf
        # print("zf:",self.zf.min().data.cpu().numpy(),self.zf.max().data.cpu().numpy(),self.zf.mean().data.cpu().numpy(),self.zf.size())

    def track(self, x):
        # print("x:",x.min().data.cpu().numpy(),x.max().data.cpu().numpy(),x.mean().data.cpu().numpy())
        x -= 114.0
        x /= 58.0
        xf = self.backbone(x)
        xf = self.neck(xf)
        # print("xfafterneck:",xf.min().data.cpu().numpy(),xf.max().data.cpu().numpy(),xf.mean().data.cpu().numpy())
        # print("self.zf:",self.zf.min().data.cpu().numpy(),self.zf.max().data.cpu().numpy(),self.zf.mean().data.cpu().numpy())

        cls, loc = self.head(self.zf,xf)
        # cls = cls.sigmoid()
        # print("cls:",cls.min().data.cpu().numpy(),cls.max().data.cpu().numpy(),cls.mean().data.cpu().numpy(),cls.size())
        # print("loc:",loc.min().data.cpu().numpy(),loc.max().data.cpu().numpy(),loc.mean().data.cpu().numpy(),loc.size())
        # exit()

        return cls,loc

    def log_softmax(self, cls):
        cls = cls.permute(0, 2, 3, 1).contiguous()
        cls = F.sigmoid(cls)
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

    def forward(self, template,search,feattea,rank=None):
        """ only used in training
        """
        # init_box = self.cornercenter(template_box)

        template = template -  114.0
        template = template / 58.0
        search = search - 114.0
        search = search/58.0

        # get feature
        zfbb = self.backbone(template)
        xfbb = self.backbone(search)

        zf = self.neck(zfbb,flagcentercrop=True)
        xf = self.neck(xfbb)

        feat = self.head(zf,xf)

        feat_loss = F.l1_loss(feat,feattea,reduction="mean")
        outputs = {}
        outputs['total_loss'] = feat_loss


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
