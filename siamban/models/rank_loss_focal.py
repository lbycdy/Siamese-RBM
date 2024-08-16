# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from siamban.core.config import cfg
from siamban.models.iou_loss import linear_iou,giou_loss
import time
import random
def IoU(rect1, rect2):
    """ caculate interection over union
    Args:
        rect1: (x1, y1, x2, y2)
        rect2: (x1, y1, x2, y2)
    Returns:
        iou
    """
    # overlap
    # print("rect1.shape:",rect1.shape)
    # print("rect2.shape:",rect2.shape)
    x1 = rect1[0]
    x2 = rect1[2]
    y1 = rect1[1]
    y2 = rect1[3]

    tx1 = rect2[0]
    tx2 = rect2[2]
    ty1 = rect2[1]
    ty2 = rect2[3]

    xx1 = torch.max(tx1, x1)
    yy1 = torch.max(ty1, y1)
    xx2 = torch.min(tx2, x2)
    yy2 = torch.min(ty2, y2)

    ww = torch.clamp((xx2 - xx1), min=0)
    hh = torch.clamp((yy2 - yy1), min=0)
    area = (x2 - x1) * (y2 - y1)
    target_a = (tx2 - tx1) * (ty2 - ty1)
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou


# def log_softmax(cls):
#     if cfg.BAN.BAN:
#         cls = cls.permute(0, 2, 3, 1).contiguous()
#         cls = F.log_softmax(cls, dim=3)
#     return cls
class Rank_CLS_Loss(nn.Module):
    def __init__(self, L=4, margin=0.5):
        super(Rank_CLS_Loss, self).__init__()
        self.margin = margin
        self.L = L

    def forward(self, input, label, rank=None,flagsigmoid=False):
        loss_all = []
        batch_size = input.shape[0]
        pred = input.view(batch_size, -1, 2)
        label = label.view(batch_size, -1)
        for batch_id in range(batch_size):
            pos_index = np.where(label[batch_id].cpu() == 1)[0].tolist()
            neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
            if len(pos_index) > 0:
                if flagsigmoid:
                    pos_prob = pred[batch_id][pos_index][:, 1]
                    neg_prob = pred[batch_id][neg_index][:, 1]
                else:
                    pos_prob = torch.exp(pred[batch_id][pos_index][:, 1])
                    neg_prob = torch.exp(pred[batch_id][neg_index][:, 1])

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
                if flagsigmoid:
                    neg_prob = pred[batch_id][neg_index][:, 1]
                else:
                    neg_prob = torch.exp(pred[batch_id][neg_index][:, 1])

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
            final_loss = torch.FloatTensor([0]).to(rank)[0]
        return final_loss

class Rank_IGR_Loss(nn.Module):
    def __init__(self):
        super(Rank_IGR_Loss, self).__init__()

    def forward(self, cls, label_cls, pred_bboxes, label_target, rank=None,flagsigmoid=False):
        batch_size = label_cls.shape[0]
        label_cls = label_cls.view(batch_size, -1)
        cls = cls.view(batch_size, -1, 2)
        loss_all_1 = []
        loss_all_2 = []
        for i in range(batch_size):
            pos_idx = label_cls[i] > 0
            num_pos = pos_idx.sum(0, keepdim=True)
            if num_pos > 0:
                if flagsigmoid:
                    pos_prob = cls[i][pos_idx][:, 1]

                else:
                    pos_prob = torch.exp(cls[i][pos_idx][:, 1])
                iou = IoU(pred_bboxes[i][:, pos_idx], label_target[i])
                iou_value, iou_idx = iou.sort(0, descending=True)
                pos_num = iou.shape[0]
                pos_num_sub_batch_size = int(pos_num * (pos_num - 1) / 2)
                input1 = torch.LongTensor(pos_num_sub_batch_size)
                input2 = torch.LongTensor(pos_num_sub_batch_size)
                index = 0
                for ii in range(pos_num - 1):
                    for jj in range((ii + 1), pos_num):
                        input1[index] = iou_idx[ii]
                        input2[index] = iou_idx[jj]
                        index = index + 1
                input1, input2 = input1.to(rank), input2.to(rank)
                loss1 = torch.exp(-cfg.TRAIN.IoU_Gamma * (pos_prob[input1] - pos_prob[input2])).mean()
                pos_prob_value, pos_prob_idx = pos_prob.sort(0, descending=True)
                pos_num = pos_prob_value.shape[0]
                pos_num_sub_batch_size = int(pos_num * (pos_num - 1) / 2)
                idx1 = torch.LongTensor(pos_num_sub_batch_size)
                idx2 = torch.LongTensor(pos_num_sub_batch_size)
                index = 0
                for ii in range(pos_num - 1):
                    for jj in range((ii + 1), pos_num):
                        idx1[index] = pos_prob_idx[ii]
                        idx2[index] = pos_prob_idx[jj]
                        index = index + 1

                idx1, idx2 = idx1.to(rank), idx2.to(rank)
                loss2 = torch.exp(-cfg.TRAIN.IoU_Gamma * (iou[idx1] - iou[idx2].detach())).mean()
                if torch.isnan(loss1) or torch.isnan(loss2):
                    continue
                else:
                    loss_all_1.append(loss1)
                    loss_all_2.append(loss2)
        if len(loss_all_1):
            final_loss1 = torch.stack(loss_all_1).mean()
        else:
            final_loss1 = torch.FloatTensor([0]).to(rank)[0]
        if len(loss_all_2):
            final_loss2 = torch.stack(loss_all_2).mean()
        else:
            final_loss2 = torch.FloatTensor([0]).to(rank)[0]
        return final_loss1, final_loss2

class Rank_IGR_Loss_FAST(nn.Module):
    def __init__(self,cfg):
        super(Rank_IGR_Loss_FAST, self).__init__()
        self.cfg = cfg


    def forward(self, cls, label_cls, pred_bboxes, label_target, rank=None,flagsigmoid=False):
        batch_size = label_cls.shape[0]
        label_cls = label_cls.view(batch_size, -1)
        cls = cls.view(batch_size, -1, 2)
        loss_all_1 = []
        loss_all_2 = []
        for i in range(batch_size):
            pos_idx = label_cls[i] > 0
            num_pos = pos_idx.sum(0, keepdim=True)
            if num_pos > 0:
                if flagsigmoid:
                    pos_prob = cls[i][pos_idx][:, 1]

                else:
                    pos_prob = torch.exp(cls[i][pos_idx][:, 1])
                iou = IoU(pred_bboxes[i][:, pos_idx], label_target[i])
                iou_value, iou_idx = iou.sort(0, descending=True)
                pos_num = iou.shape[0]
                if pos_num > self.cfg.TRAIN.NPOS_SEL_RANK:
                    idxsel = torch.tensor(random.sample(list(range(pos_num)), self.cfg.TRAIN.NPOS_SEL_RANK))
                    idxsel = idxsel.sort()[0]
                    iou_idx = iou_idx[idxsel]
                    pos_num = iou_idx.shape[0]

                pos_num_sub_batch_size = int(pos_num * (pos_num - 1) / 2)
                input1 = torch.LongTensor(pos_num_sub_batch_size)
                input2 = torch.LongTensor(pos_num_sub_batch_size)
                index = 0
                for ii in range(pos_num - 1):
                    for jj in range((ii + 1), pos_num):
                        input1[index] = iou_idx[ii]
                        input2[index] = iou_idx[jj]
                        index = index + 1
                input1, input2 = input1.to(rank), input2.to(rank)
                loss1 = torch.exp(-cfg.TRAIN.IoU_Gamma * (pos_prob[input1] - pos_prob[input2])).mean()
                pos_prob_value, pos_prob_idx = pos_prob.sort(0, descending=True)
                pos_num = pos_prob_value.shape[0]
                if pos_num > self.cfg.TRAIN.NPOS_SEL_RANK:
                    idxsel = torch.tensor(random.sample(list(range(pos_num)), self.cfg.TRAIN.NPOS_SEL_RANK))
                    idxsel = idxsel.sort()[0]
                    pos_prob_idx = pos_prob_idx[idxsel]
                    pos_num = pos_prob_idx.shape[0]
                pos_num_sub_batch_size = int(pos_num * (pos_num - 1) / 2)
                idx1 = torch.LongTensor(pos_num_sub_batch_size)
                idx2 = torch.LongTensor(pos_num_sub_batch_size)
                index = 0
                for ii in range(pos_num - 1):
                    for jj in range((ii + 1), pos_num):
                        idx1[index] = pos_prob_idx[ii]
                        idx2[index] = pos_prob_idx[jj]
                        index = index + 1

                idx1, idx2 = idx1.to(rank), idx2.to(rank)
                loss2 = torch.exp(-cfg.TRAIN.IoU_Gamma * (iou[idx1] - iou[idx2].detach())).mean()
                if torch.isnan(loss1) or torch.isnan(loss2):
                    continue
                else:
                    loss_all_1.append(loss1)
                    loss_all_2.append(loss2)
        if len(loss_all_1):
            final_loss1 = torch.stack(loss_all_1).mean()
        else:
            final_loss1 = torch.FloatTensor([0]).to(rank)[0]
        if len(loss_all_2):
            final_loss2 = torch.stack(loss_all_2).mean()
        else:
            final_loss2 = torch.FloatTensor([0]).to(rank)[0]
        return final_loss1, final_loss2

def centerness_distance(label_loc, label_target):
    # x1 = label_loc[0]
    # x2 = label_loc[2]
    # y1 = label_loc[1]
    # y2 = label_loc[3]
    #
    # cx = (x1 + x2) / 2
    # cy = (y1 + y2) / 2
    cx = label_loc[0] + label_target[0]
    cy = label_loc[1] + label_target[1]

    tx1 = label_target[0]
    tx2 = label_target[2]
    ty1 = label_target[1]
    ty2 = label_target[3]

    t_cx = (tx1 + tx2) / 2
    t_cy = (ty1 + ty2) / 2

    return pow(pow(cx-t_cx,2)+pow(cy-t_cy,2),0.5)

def del_tensor(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2),dim=0)

class Rank_IGR_Centerness_Loss(nn.Module):
    def __init__(self, cfg=None):
        super(Rank_IGR_Centerness_Loss, self).__init__()
        self.cfg = cfg

    def forward(self, cls, label_cls, label_loc, pred_bboxes, label_target, flagsigmoid,rank=None):
        batch_size = label_cls.shape[0]
        label_cls = label_cls.view(batch_size, -1)
        label_loc = label_loc.view(batch_size, 4, -1)

        left_right = label_loc[:, [0, 2]]
        top_bottom = label_loc[:, [1, 3]]
        centerness = (left_right.min(dim=1)[0] / left_right.max(dim=1)[0]) * \
                     (top_bottom.min(dim=1)[0] / top_bottom.max(dim=1)[0])
        centerness= torch.sqrt(centerness)

        cls = cls.view(batch_size, -1, 2)
        loss_all_1 = []
        loss_all_2 = []
        # distance_ratio=0
        for i in range(batch_size):
            pos_idx = label_cls[i] > 0
            num_pos = pos_idx.sum(0, keepdim=True)
            if num_pos > 0:
                if flagsigmoid:
                    pos_prob = cls[i][pos_idx][:, 1]
                else:
                    pos_prob = torch.exp(cls[i][pos_idx][:, 1])
                iou = IoU(pred_bboxes[i][:, pos_idx], label_target[i])
                # center_distance = centerness_distance(label_loc[i][:, pos_idx], label_target[i])
                center_distance = centerness[i][pos_idx]

                distance, dis_idx = center_distance.sort(0,descending=True)

                pos_num = distance.shape[0]
                if pos_num>self.cfg.TRAIN.NPOS_SEL_RANK:
                    idxsel = torch.tensor(random.sample(list(range(pos_num)), self.cfg.TRAIN.NPOS_SEL_RANK))
                    idxsel = idxsel.sort()[0]
                    distance = distance[idxsel]
                    dis_idx = dis_idx[idxsel]
                    pos_num = distance.shape[0]

                pos_num_sub_batch_size = int(pos_num * (pos_num - 1) / 2)

                idx1 = torch.LongTensor(pos_num_sub_batch_size)
                idx2 = torch.LongTensor(pos_num_sub_batch_size)
                index1 = 0
                # t1 = time.time()
                for ii in range(pos_num - 1):
                    for jj in range((ii+1), pos_num):
                        if abs(distance[jj] - distance[ii]) > 0.05:
                            idx1[index1] = dis_idx[ii]
                            idx2[index1] = dis_idx[jj]
                            index1 = index1 + 1
                idx1 = idx1[:index1]
                idx2 = idx2[:index1]

                t2 = time.time()
                # print(i,pos_num,t2-t1)
                if rank is None:
                    idx1, idx2 = idx1.cuda(), idx2.cuda()
                else:
                    idx1, idx2 = idx1.to(rank),idx2.to(rank)
                loss1 = torch.exp(-self.cfg.TRAIN.IoU_Gamma1 * (pos_prob[idx1] - pos_prob[idx2])).mean()
                loss2 = torch.exp(-self.cfg.TRAIN.IoU_Gamma2 * (iou[idx1] - iou[idx2].detach())).mean()

                if torch.isnan(loss1) or torch.isnan(loss2):
                    continue
                else:
                    loss_all_1.append(loss1)
                    loss_all_2.append(loss2)
        if len(loss_all_1):
            final_loss1 = torch.stack(loss_all_1).mean()
        else:
            final_loss1 = torch.FloatTensor([0]).cuda()[0]
        if len(loss_all_2):
            final_loss2 = torch.stack(loss_all_2).mean()
        else:
            final_loss2 = torch.FloatTensor([0]).cuda()[0]
        # print("distance_ratio", distance_ratio/batch_size)
        return final_loss1, final_loss2

def get_cls_loss(pred, label, select,flagsigmoid=False,flaguselogits=False):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    if flagsigmoid:
        label = torch.stack([1.0-label,label],dim=1).float()
        if flaguselogits:
            return F.binary_cross_entropy_with_logits(pred,label)
        else:
            return F.binary_cross_entropy(pred,label)
    else:
        return F.nll_loss(pred, label)

def select_cross_entropy_loss(pred, label,flagsigmoid=False,rank=None,flaguselogits=False):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().to(rank)
    neg = label.data.eq(0).nonzero().squeeze().to(rank)
    loss_pos = get_cls_loss(pred, label, pos,flagsigmoid,flaguselogits)
    loss_neg = get_cls_loss(pred, label, neg,flagsigmoid,flaguselogits)
    return loss_pos * 0.5 + loss_neg * 0.5

def select_focal_loss(pred, label,rank=None,alpha=0.25, gamma = 2):
    pred = pred.view(-1, 2)[:,1]
    label = label.view(-1).float()
    idxes = label.data.ge(0).nonzero().squeeze().to(rank)
    inputs = torch.index_select(pred, 0, idxes)
    targets = torch.index_select(label, 0, idxes)
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t + 1e-6) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    loss = loss.mean()
    return loss

def weight_l1_loss(pred_loc, label_loc, loss_weight):
    if cfg.BAN.BAN:
        diff = (pred_loc - label_loc).abs()
        diff = diff.sum(dim=1)
    else:
        diff = None
    loss = diff * loss_weight
    return loss.sum().div(pred_loc.size()[0])

def select_iou_loss(pred_loc, label_loc, label_cls, rank=None,losstype='linear_iou'):
    label_cls = label_cls.reshape(-1)
    pos = label_cls.data.eq(1).nonzero().squeeze().to(rank)

    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)

    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)
    if losstype=='linear_iou':
        return linear_iou(pred_loc, label_loc)
    elif losstype=='giou':
        return giou_loss(pred_loc, label_loc)

def select_l1_loss(pred_loc, label_loc, label_cls, rank=None):
    label_cls = label_cls.reshape(-1)
    pos = label_cls.data.eq(1).nonzero().squeeze().to(rank)

    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)

    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1)
    return diff.sum().div(pred_loc.size()[0])

def select_iou_loss_multiclass(pred_loc, label_loc, label_cls, rank=None):
    label_cls = label_cls.reshape(-1)
    pos = label_cls.data.gt(0).nonzero().squeeze().to(rank)

    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)

    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)

    return linear_iou(pred_loc, label_loc)

def rank_cls_loss():
    loss = Rank_CLS_Loss()
    return loss

def rank_loc_loss():
    loss = Rank_IGR_Loss()
    return loss

def rank_center_loc_loss(cfg):
    loss = Rank_IGR_Centerness_Loss(cfg)
    return loss

def rank_loc_loss_fast(cfg):
    loss = Rank_IGR_Loss_FAST(cfg)
    return loss

def segmentation_loss(pred_seg, true_seg,flaglogits=True):
    """
    Args:
        pred_seg: Shape(B, T, 1, H, W)
        true_seg: Shape(B, T, 1, H, W)
    """
    return F.binary_cross_entropy_with_logits(pred_seg, true_seg)