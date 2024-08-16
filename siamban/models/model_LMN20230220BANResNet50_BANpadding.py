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
import math
def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel, padding=2)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = l + 7
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer_RELU(nn.Module):
    def __init__(self, in_channels, out_channels, model_paraminit='aaa'):
        super(AdjustAllLayer_RELU, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i]))

        if model_paraminit == 'uniform':
            self._init_weights_uniform()
        elif model_paraminit == 'gauss':
            self._init_weights_gauss()
        elif model_paraminit == 'gauss1':
            self._init_weights_gauss1()
        elif model_paraminit == 'normal':
            self._init_weights_noraml()

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out

    def _init_weights_noraml(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.in_channels
                for k in m.kernel_size:
                    n *= k
                stdv = 1. / math.sqrt(n)
                m.weight.data.normal_(-stdv, stdv)  # Error not the same as torch0.4.1
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _init_weights_uniform(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.in_channels
                for k in m.kernel_size:
                    n *= k
                stdv = 1. / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def _init_weights_gauss(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def _init_weights_gauss1(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, relu=True):
        super(DepthwiseXCorr, self).__init__()
        if relu:
            self.conv_kernel = nn.Sequential(
                    nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(inplace=True),
                )
        elif not relu:
            self.conv_kernel = nn.Sequential(
                    nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                    nn.BatchNorm2d(hidden),
                )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        # print(kernel.size(),search.size())
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        # print(kernel.size(),search.size(),feature.size())
        # print(kernel.min().data.cpu().numpy(),kernel.max().data.cpu().numpy(),kernel.abs().sum().data.cpu().numpy(),search.min().data.cpu().numpy(),search.max().data.cpu().numpy(),search.sum().data.cpu().numpy(),feature.min().data.cpu().numpy(),feature.max().data.cpu().numpy())
        out = self.head(feature)
        return out,feature


class DepthwiseBAN(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, cls_out_channels=2, weighted=False, relu=True):
        super(DepthwiseBAN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, cls_out_channels, relu=relu)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4, relu=relu)

    def forward(self, z_f, x_f):
        cls,featcls = self.cls(z_f, x_f)
        loc,featloc = self.loc(z_f, x_f)
        return cls, loc,featcls,featloc


class MultiBAN_PADDING(nn.Module):
    def __init__(self, in_channels, cls_out_channels, weighted=False, relu=True, model_paraminit='aaa'):
        super(MultiBAN_PADDING, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('box' + str(i + 2), DepthwiseBAN(in_channels[i], in_channels[i], cls_out_channels, relu=relu))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
        self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))
        if model_paraminit == 'uniform':
            self._init_weights_uniform()
        elif model_paraminit == 'gauss':
            self._init_weights_gauss()
        elif model_paraminit == 'gauss1':
            self._init_weights_gauss1()
        elif model_paraminit == 'normal':
            self._init_weights_noraml()

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        featcls = []
        featloc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            box = getattr(self, 'box' + str(idx))
            c, l,fc,fl = box(z_f, x_f)
            cls.append(c)
            loc.append(torch.exp(l * self.loc_scale[idx - 2]))
            featcls.append(fc)
            featloc.append(fl)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)
            # print("loc_weight:",cls_weight,loc_weight)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight),featcls,featloc
        else:
            return avg(cls), avg(loc),featcls,featloc

    def _init_weights_noraml(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.in_channels
                for k in m.kernel_size:
                    n *= k
                stdv = 1. / math.sqrt(n)
                m.weight.data.normal_(-stdv, stdv)  # Error not the same as torch0.4.1
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _init_weights_uniform(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.in_channels
                for k in m.kernel_size:
                    n *= k
                stdv = 1. / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def _init_weights_gauss(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def _init_weights_gauss1(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        # build backbone
        #print(cfg.BACKBONE.TYPE)
        self.backbone = resnet50(used_layers=[2, 3, 4])

        # build adjust layer
        self.neck = AdjustAllLayer_RELU(in_channels=[512, 1024, 2048],out_channels=[256, 256, 256])

        # build ban head
        self.head = MultiBAN_PADDING(in_channels=[256, 256, 256],cls_out_channels=2,weighted=True)

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

    def forward(self, data,rank=None):
        """ only used in training
        """
        # template = data['template'].cuda()
        # search = data['search'].cuda()
        # label_cls = data['label_cls'].cuda()
        # label_loc = data['label_loc'].cuda()
        if rank is None:
            template = data['template'].cuda()
            search = data['search'].cuda()
        else:
            template = data['template'].to(rank)
            search = data['search'].to(rank)

        # template_box = data['template_box'].cuda()
        # init_box = self.cornercenter(template_box)

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        zf = self.neck(zf)
        xf = self.neck(xf)
        cls, loc,featcls,featloc = self.head(zf, xf)

        return cls,loc,featcls,featloc
