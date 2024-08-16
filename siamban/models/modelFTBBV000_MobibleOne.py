# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

#from siamban.core.config import cfg
from siamban.models.rank_loss import  rank_cls_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck
from siamban.utils.point import Point
from siamban.models.backbone.mobileone_stride import mobileone
from siamban.core.config import cfg
import numpy as np



class Score(nn.Module):
    def __init__(self, in_channels, hidden, kernel_size=7):
        super(Score, self).__init__()
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

    def forward(self, kernel,search):
        kernel = kernel[:,:,4:11,4:11]
        conv_kernel = self.conv_kernel(kernel)
        conv_search = self.conv_search(search)
        k_norm = torch.norm(conv_kernel, p=2, dim=1, keepdim=True)
        s_norm = torch.norm(conv_search, p=2, dim=1, keepdim=True)
        conv_kernel = conv_kernel/(k_norm+1e-8)
        conv_search = conv_search/(s_norm+1e-8)
        score = conv_kernel * conv_search
        score = score.sum(dim=1, keepdim=True)
        return score


class MultiScore(nn.Module):
    def __init__(self, in_channels=[], hidden=256):
        super(MultiScore, self).__init__()
        for i in range(len(in_channels)):
            self.add_module('score'+str(i+2), Score(in_channels[i], hidden))
        self.weight = nn.Parameter(torch.ones(len(in_channels)))
    def forward(self, z_fs,x_fs):
        s_list = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            score = getattr(self, 'score'+str(idx))
            si = score(z_f, x_f)
            s_list.append(si)

        weight = F.softmax(self.weight, 0)
        s = 0
        for i in range(len(weight)):
            s += s_list[i] * weight[i]
        return s

def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.binary_cross_entropy(pred,label)
def select_cross_entropy_loss(pred, label, rank=None):
    pred = pred.view(-1)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().to(rank)
    neg = label.data.eq(0).nonzero().squeeze().to(rank)
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5

class Rank_CLS_Loss(nn.Module):
    def __init__(self, L=4, margin=0.5):
        super(Rank_CLS_Loss, self).__init__()
        self.margin = margin
        self.L = L

    def forward(self, input, label, rank=None):
        loss_all = []
        batch_size = input.shape[0]
        pred = input.view(batch_size, -1, 1)
        label = label.view(batch_size, -1)
        for batch_id in range(batch_size):
            pos_index = np.where(label[batch_id].cpu() == 1)[0].tolist()
            neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
            if len(pos_index) > 0:
                pos_prob = pred[batch_id][pos_index][:, 0]
                neg_prob = pred[batch_id][neg_index][:, 0]

                num_pos = len(pos_index)
                neg_value, _ = neg_prob.sort(0, descending=True)
                pos_value, _ = pos_prob.sort(0, descending=True)
                neg_idx2 = neg_prob > cfg.TRAIN.HARD_NEGATIVE_THS
                if neg_idx2.sum() == 0:
                    continue
                neg_value = neg_value[0:num_pos]

                pos_value = pos_value[0:num_pos]
                neg_q = F.softmax(neg_value, dim=0)
                neg_dist = torch.sum(neg_value * neg_q)

                pos_dist = torch.sum(pos_value) / len(pos_value)
                loss = torch.log(1. + torch.exp(self.L * (neg_dist - pos_dist + self.margin))) / self.L
            else:
                neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
                neg_prob = pred[batch_id][neg_index][:, 0]

                neg_value, _ = neg_prob.sort(0, descending=True)
                neg_idx2 = neg_prob > cfg.TRAIN.HARD_NEGATIVE_THS
                if neg_idx2.sum() == 0:
                    continue
                num_neg = len(neg_prob[neg_idx2])
                num_neg = max(num_neg, cfg.TRAIN.RANK_NUM_HARD_NEGATIVE_SAMPLES)
                neg_value = neg_value[0:num_neg]
                neg_q = F.softmax(neg_value, dim=0)
                neg_dist = torch.sum(neg_value * neg_q)
                loss = torch.log(1. + torch.exp(self.L * (neg_dist - 1. + self.margin))) / self.L

            loss_all.append(loss)
        if len(loss_all):
            final_loss = torch.stack(loss_all).mean()
        else:
            final_loss = torch.zeros(1).to(rank)

        return final_loss

class ModelBuilder(nn.Module):
    def __init__(self,cfg):
        super(ModelBuilder, self).__init__()

        self.cfg = cfg
        self.backbone = mobileone(variant='s0')

        # build adjust layer
        self.neck = MultiScore(in_channels=[128,256,1024],hidden=256)
        self.rank_cls_loss=Rank_CLS_Loss()

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
            label_cls = data['label_cls'].cuda().float()
        else:
            template = data['template'].to(rank)
            search = data['search'].to(rank)
            label_cls = data['label_cls'].to(rank).float()

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

        score = self.neck(zf,xf)
        cls_loss = select_cross_entropy_loss(score, label_cls, rank=rank)

        # loc loss with iou loss

        CR_loss=self.rank_cls_loss(score,label_cls, rank=rank)
        outputs = {}
        outputs['total_loss'] = self.cfg.TRAIN.CLS_WEIGHT * cls_loss  + self.cfg.TRAIN.RANK_CLS_WEIGHT*CR_loss
        outputs['cls_loss'] = cls_loss
        outputs['CR_loss'] = self.cfg.TRAIN.RANK_CLS_WEIGHT*CR_loss

        return outputs
