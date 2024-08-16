# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self,cfg=None):
        super(ModelBuilder, self).__init__()
        self.bbox = bbox = None

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

    def template(self, z):
        # self.bbox = bbox
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf
        self.head.init(zf)

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.head.track(xf)
        return {
                'cls': cls,
                'loc': loc
               }

    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            # cls = F.log_softmax(cls, dim=3)
            if cfg.TRAIN.FLAG_SIGMOID_LOSS:
                # cls = F.logsigmoid(cls)
                cls = F.sigmoid(cls)
            else:
                cls = F.log_softmax(cls, dim=3)
        return cls

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
            label_cls = data['label_cls'].cuda()
            label_loc = data['label_loc'].cuda()
        else:
            template = data['template'].to(rank)
            search = data['search'].to(rank)
            label_cls = data['label_cls'].to(rank)
            label_loc = data['label_loc'].to(rank)

        # template_box = data['template_box'].cuda()
        # init_box = self.cornercenter(template_box)

        if cfg.TRAIN.DATA_NORMALIZE:
            if cfg.TRAIN.DATA_NORMALIZE_MODE == 0:
                template /= 255.0
                search /= 255.0
                template -= self.mean
                template /= self.std
                search -= self.mean
                search /= self.std
            elif cfg.TRAIN.DATA_NORMALIZE_MODE == 1:
                template -= 114.0
                template /= 58.0
                search -= 114.0
                search /= 58.0

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        # self.head.init(zf, init_box)
        self.head.init(zf)
        cls, loc = self.head.track(xf)

        # get loss

        # cls loss with cross entropy loss
        cls = self.log_softmax(cls)
        # cls_loss = select_cross_entropy_loss(cls, label_cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls, cfg.TRAIN.FLAG_SIGMOID_LOSS)

        # loc loss with iou loss
        loc_loss = select_iou_loss(loc, label_loc, label_cls)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss


        return outputs
