# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
from siamban.utils.bbox import center2corner, Center,corner2center
from siamban.core.config import cfg
from siamban.models.rank_loss import select_cross_entropy_loss, select_iou_loss, rank_cls_loss,rank_loc_loss

from siamban.models.backbone.mobileone_stride import mobileone
from siamban.models.backbone.mobileone_strideS16OutTwo import mobileone as mobileones16outtwo

import torch
import torchvision
import math
import numpy as np
import random




class Anchors:
    """
    This class generate anchors.
    """
    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = image_center
        self.size = size

        self.anchor_num = len(self.scales) * len(self.ratios)

        self.anchors = None

        self.generate_anchors()

    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size*1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w*0.5, -h*0.5, w*0.5, h*0.5][:]
                count += 1

    def generate_all_anchors(self, im_c, size):
        """
        im_c: image center
        size: image size
        """
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size
        #im_c = 255/2, size = 25,stride = 8
        a0x = im_c - size // 2 * self.stride
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori

        x1 = zero_anchors[:, 0]
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]
        # print(zero_anchors)
        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1),
                             [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])

        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x
        cy = cy + disp_y

        # broadcast
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])

        self.all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                            np.stack([cx, cy, w,  h]).astype(np.float32))
        # print(self.all_anchors[0].shape,self.all_anchors[1].shape)
        # print(self.all_anchors[0][:,0,2,2])
        return self.all_anchors

class AnchorTarget:
    def __init__(self,):
        self.anchors = Anchors(cfg.ANCHOR.STRIDE,
                               cfg.ANCHOR.RATIOS,
                               cfg.ANCHOR.SCALES)

        self.all_anchors = self.anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE//2,
                                          size=cfg.TRAIN.OUTPUT_SIZE)


class IOULoss1(nn.Module):
    def  __init__(self, loc_loss_type):
        super(IOULoss1, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)
        target_area = (target_right - target_left) * (target_bottom - target_top)


        w_intersect = torch.min(pred_right, target_right) - torch.max(pred_left, target_left)
        w_intersect = torch.clamp(w_intersect, min=0)
        h_intersect = torch.min(pred_bottom, target_bottom) - torch.max(pred_top, target_top)
        h_intersect = torch.clamp(h_intersect, min=0)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)


        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

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
            nn.Conv2d(feature_in, 2 * cfg.ANCHOR.ANCHOR_NUM, kernel_size=3, bias=True, padding=1),
        )
        self.loc = nn.Sequential(
            nn.Conv2d(feature_in, feature_in, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(feature_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in, feature_in, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(feature_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in, 4 * cfg.ANCHOR.ANCHOR_NUM, kernel_size=3, bias=True, padding=1),
        )

        self.loc_scale = nn.Parameter(torch.ones(1))

    def forward(self, xf,zf):
        depthcorr = self.depthcorr(xf,zf)
        head1 = self.head1(depthcorr)
        cls = self.cls(head1)
        loc = self.loc(head1)
        return cls,loc

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
def select_iou_loss1(pred_loc, label_loc, label_cls, rank=None):

    label_cls = label_cls.reshape(-1)    #(80,6,19,19)
    if rank is None:
        pos = label_cls.data.eq(1).nonzero().squeeze().cuda()
    else:
        pos = label_cls.data.eq(1).nonzero().squeeze().to(rank, non_blocking=True)

    pred_loc = torch.index_select(pred_loc, 0, pos)
    label_loc = torch.index_select(label_loc, 0, pos)
    linear_iou1 = IOULoss1(loc_loss_type='linear_iou')
    loss = linear_iou1(pred_loc, label_loc)
    return loss
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
    def log_softmax1(self, cls):  # (80, 12, 11, 11)
        b, _, h, w = cls.size()
        cls = cls.view(b, 2, -1, h, w)# (80, 2, 6, 11, 11)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()# (80, 6, 11, 11, 2)
        if cfg.TRAIN.FLAG_SIGMOID_LOSS:
            cls = F.sigmoid(cls)
        else:
            cls = F.log_softmax(cls, dim=4)
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
        label_target /= (float(imgsize)/10.0)
        label_loc /= (float(imgsize)/10.0)
        anchortarget = AnchorTarget()
        anchor_center = anchortarget.all_anchors[1]
        anchor_center = torch.from_numpy(anchor_center).to(rank).view(4, -1)

        label_cls = torch.tensor(label_cls)
        # point_tensor = torch.from_numpy(self.points.points).to(rank).view(1, 2, -1)/(float(imgsize)/10.0)
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
        delta = loc.view(Nimg, 4, -1,19 ,19).reshape(Nimg, 4, -1) # delta
        pred_bboxes_cen = delta.clone()
        pred_bboxes = torch.zeros_like(delta)
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]


        pred_bboxes_cen[:, 0, :] = delta[:, 0, :] * w + cx
        pred_bboxes_cen[:, 1, :] = delta[:, 1, :] * h + cy
        pred_bboxes_cen[:, 2, :] = torch.exp(delta[:, 2, :]) * w
        pred_bboxes_cen[:, 3, :] = torch.exp(delta[:, 3, :]) * h
        pred_bboxes[:, :2, :] = pred_bboxes_cen[:, :2, :] - 1 / 2 * pred_bboxes_cen[:, 2:, :]
        pred_bboxes[:, 2:, :] = pred_bboxes_cen[:, :2, :] + 1 / 2 * pred_bboxes_cen[:, 2:, :]
        pred_bboxes = pred_bboxes.permute(0, 2, 1).flatten(0, 1)  # (80, 1089,4)
        pred_bboxes /= (float(imgsize) / 10.0)

        label_loc *= (float(imgsize) / 10.0)
        label_loc = label_loc.flatten(2, 4)
        gt_bboxes_cen = label_loc.clone()
        gt_bboxes = torch.zeros_like(label_loc)
        gt_bboxes_cen[:, 0, :] = gt_bboxes_cen[:, 0, :] * w + cx
        gt_bboxes_cen[:, 1, :] = gt_bboxes_cen[:, 1, :] * h + cy
        gt_bboxes_cen[:, 2, :] = torch.exp(gt_bboxes_cen[:, 2, :]) * w
        gt_bboxes_cen[:, 3, :] = torch.exp(gt_bboxes_cen[:, 3, :]) * h
        gt_bboxes[:, :2, :] = gt_bboxes_cen[:, :2, :] - 1 / 2 * gt_bboxes_cen[:, 2:, :]
        gt_bboxes[:, 2:, :] = gt_bboxes_cen[:, :2, :] + 1 / 2 * gt_bboxes_cen[:, 2:, :]
        gt_bboxes = gt_bboxes.permute(0, 2, 1).flatten(0, 1)
        gt_bboxes /= (float(imgsize) / 10.0)


        cls = self.log_softmax1(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls, cfg.TRAIN.FLAG_SIGMOID_LOSS,rank)
        loc_loss = select_iou_loss1(pred_bboxes, gt_bboxes, label_cls, rank=rank)
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss.detach()
        outputs['loc_loss'] = loc_loss.detach()
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