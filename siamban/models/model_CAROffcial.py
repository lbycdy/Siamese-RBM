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
from siamban.models.backbone.resnet_atrous import resnet50

from siamban.utils.point import Point
from siamban.core.xcorr import xcorr_fast, xcorr_depthwise


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = l + 7
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        # build backbone
        #print(cfg.BACKBONE.TYPE)
        self.backbone = resnet50(used_layers=[2, 3, 4])

        # build adjust layer
        self.neck = AdjustAllLayer(in_channels=[512, 1024, 2048],out_channels=[256, 256, 256])
        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)


    def template(self, z):
        # self.bbox = bbox
        zf = self.backbone(z)
        if self.cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf
        self.head.init(zf)

    def track(self, x):
        xf = self.backbone(x)
        if self.cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.head.track(xf)
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

    def forward(self, template,search,rank=None):
        """ only used in training
        """

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        zf = self.neck(zf)
        xf = self.neck(xf)
        features = xcorr_depthwise(xf[0], zf[0])
        for i in range(len(xf) - 1):
            features_new = xcorr_depthwise(xf[i + 1], zf[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.down(features)

        return features
