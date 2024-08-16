# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops

from siamban.core.config import cfg
from siamban.models.rank_loss import select_cross_entropy_loss, select_iou_loss, rank_cls_loss,rank_loc_loss,select_iou_loss_multiclass
from siamban.models.loss_multiobj import FCOSAlikeLossComputation

from siamban.models.backbone.mobileone_stride import mobileone
from siamban.models.backbone.mobileone_strideS16OutTwo import mobileone as mobileones16outtwo


# from siamban.utils.structures.bounding_box import BoxList
# from siamban.utils.structures.boxlist_ops import cat_boxlist
# from siamban.utils.structures.boxlist_ops import boxlist_ml_nms
# from siamban.utils.structures.boxlist_ops import remove_small_boxes

import torch
import torchvision
import math
import numpy as np
import random

def reduce_sum(tensor,numgpus):
    if numgpus <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor

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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, hidden=64):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, hidden, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(hidden, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        return self.sigmoid(avg_out)

class AdjustAllLayerFPNAddFlagCenterCrop(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayerFPNAddFlagCenterCrop, self).__init__()
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels[1], 1, bias=False),
            nn.BatchNorm2d(out_channels[1]),
        )
        self.ca1 = ChannelAttention(out_channels[0], 128)
        self.ca2 = ChannelAttention(out_channels[1], 128)

    def forward(self, features,flagcentercrop=False):
        downsample1 = self.downsample1(features[0])
        downsample2 = self.downsample2(features[1])
        downsample1 = self.ca1(downsample1) * downsample1
        downsample2 = self.ca2(downsample2) * downsample2

        out = downsample1+downsample2
        return out


class RPNHead(nn.Module):
    def __init__(self,hidden=192):
        super(RPNHead, self).__init__()

        self.head1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(hidden, hidden,kernel_size=3, bias=False,padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=3, bias=True, padding=1),
        )
        self.loc = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 4, kernel_size=3, bias=True, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)


    def forward(self, featbb,sim):
        feat = featbb*sim
        head1 = self.head1(feat)
        cls = self.cls(head1)
        loc = self.loc(head1)
        return cls,self.relu(loc)


class FPN(nn.Module):
    def __init__(self,cin = [],hidden=192):
        super(FPN, self).__init__()
        self.adap1 = nn.Sequential(
            nn.Conv2d(cin[0], hidden, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(hidden)
        )
        self.adap2 = nn.Sequential(
            nn.Conv2d(cin[1], hidden, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(hidden)
        )

    def forward(self, featbb):
        adap1 = self.adap1(featbb[0])
        adap2 = self.adap2(featbb[1])
        feat = adap1+adap2
        return feat

class DetHead(nn.Module):
    def __init__(self,chan_in,hidden,numclasses):
        super(DetHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(chan_in, hidden, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.cls =  nn.Conv2d(hidden, numclasses, kernel_size=3, bias=True, padding=1)
        self.loc = nn.Conv2d(hidden, 4, kernel_size=3, bias=True, padding=1)

    def forward(self, feature):
        head = self.head(feature)
        cls = self.cls(head)
        loc = self.loc(head)
        return cls,torch.exp(loc)

class ModelBuilder(nn.Module):
    def __init__(self,cfg=None):
        super(ModelBuilder, self).__init__()
        outch = cfg.BAN.channels
        # build backbone
        self.backbone = mobileones16outtwo(variant='s0')
        # build adjust layer
        self.neck = AdjustAllLayerFPNAddFlagCenterCrop(in_channels=[256,1024],out_channels=[outch,outch])
        self.fpn = FPN([128,256],outch)
        # build rpn head
        self.head = RPNHead(outch)
        self.cls_temp = nn.Conv2d(outch, cfg.TRAIN.NUM_CLASSES, kernel_size=1, bias=True, padding=0)
        self.cnt = 0



    def template(self, z):
        z -= 114.0
        z /= 58.0
        yct = xct = 7
        zfbb = self.backbone(z)
        zf = self.neck(zfbb[1:])
        clszf = self.cls_temp(zf)
        clszf_sig = clszf.sigmoid()
        N,C = clszf_sig.size()[:2]
        clszf_center = clszf_sig[:, :, yct, xct].view((1, C, 1, 1))  #
        self.zf = clszf_center
        cls = clszf_center.data.cpu().numpy().squeeze()
        cls.sort()
        print(cls)
    def track(self, x):
        # import numpy as np
        # x = np.load("/home/ethan/SGS_IPU_SDK_v1.1.6/x_crop.npy").transpose((0, 3, 1, 2))
        # x = torch.from_numpy(x).cuda()
        # print(x.size(),"xxxx")
        imgsize = x.size()[2]
        x -= 114.0
        x /= 58.0
        xfbb = self.backbone(x)
        xf = self.neck(xfbb[1:])
        clsxf = self.cls_temp(xf)
        clsxf_sig = clsxf.sigmoid()
        sim = torch.abs(self.zf - clsxf_sig).sum(dim=1, keepdim=True).sigmoid()
        fpnxf = self.fpn(xfbb[:2])
        cls, loc = self.head(fpnxf, sim)
        s = (float(imgsize) / 10.0)
        loc *= s
        return cls,loc,sim



    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        cls = cls.permute(0, 2, 3, 1).contiguous()
        cls = F.sigmoid(cls)
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


    def forward(self, data,rank=None,numgpus=1):
        """ only used in training
        """
        Nbatch, Nt = data['template'].size()[:2]
        Nbatch, Ns = data['search'].size()[:2]
        if rank is None:
            template = data['template'].cuda().float().flatten(0,1)
            search = data['search'].cuda().float().flatten(0,1)
            template_cid = data['template_cid'].cuda().long().flatten(0,1)
            search_poscid = data['search_poscid'].cuda().long().flatten(0,1)
            label_cls_corr = data['cls_corr'].cuda().float().flatten(0,1)
            delta_corr = data['delta_corr'].cuda().float().flatten(0,1)

        else:
            template = data['template'].float().to(rank, non_blocking=True).flatten(0,1)
            search = data['search'].float().to(rank, non_blocking=True).flatten(0,1)
            template_cid = data['template_cid'].long().to(rank, non_blocking=True).flatten(0,1)
            search_poscid = data['search_poscid'].long().to(rank, non_blocking=True).flatten(0,1)
            label_cls_corr = data['cls_corr'].float().to(rank, non_blocking=True).flatten(0,1)
            delta_corr = data['delta_corr'].float().to(rank, non_blocking=True).flatten(0,1)


        # print(template.size(),search.size(),label_cls.size(),label_loc.size(),label_target.size())
        # print(torch.unique(label_cls))
        # print(template.size(),search.size(),":tempsearch")
        imgsize = search.size()[2]
        delta_corr /= (float(imgsize) / 10.0)
        xct,yct = 7,7
        template -= 114.0
        template /= 58.0
        search -= 114.0
        search /= 58.0
        # get feature
        zfbb = self.backbone(template)
        xfbb = self.backbone(search)
        zf = self.neck(zfbb[1:])
        xf = self.neck(xfbb[1:])
        clszf = self.cls_temp(zf)
        clsxf = self.cls_temp(xf)
        clszf_sig = clszf.sigmoid()
        clsxf_sig = clsxf.sigmoid()
        N,C = clszf_sig.size()[:2]
        clszf_center = clszf_sig[:,:,yct,xct].view((N,C,1,1))#

        cls_feat_temp = clszf[:,:,yct,xct]

        cls_feat_search = []
        cls_label = []
        for ib in range(N):
            for j in range(search_poscid.size()[1]):
                if search_poscid[ib,j,0]!=-1:
                    xc,yc = search_poscid[ib,j,1],search_poscid[ib,j,2]
                    cls_feat_search.append(clsxf[ib,:,yc,xc])
                    cls_label.append(search_poscid[ib,j,0])
        cls_feat_search = torch.stack(cls_feat_search)
        cls_feat = torch.cat([cls_feat_temp,cls_feat_search],dim=0)
        cls_label = torch.stack(cls_label).long()
        cls_label = torch.cat([template_cid,cls_label],dim=0)
        cls_label = F.one_hot(cls_label, num_classes=cfg.TRAIN.NUM_CLASSES + 1).float()[:, 1:]
        # cls_loss = F.binary_cross_entropy_with_logits(cls_feat, cls_label, reduction='mean')
        # num_pos_avg_per_gpu = max(total_num_pos / float(numgpus), 1.0)
        num_pos_avg_per_gpu = cls_feat.size()[0]
        cls_loss = tvops.sigmoid_focal_loss(cls_feat, cls_label, alpha=cfg.TRAIN.FOCAL_ALPHA, gamma=cfg.TRAIN.FOCAL_GAMA,
                                                reduction='sum')/num_pos_avg_per_gpu

        clszf_center = torch.unsqueeze(clszf_center, dim=1)
        clszf_center = torch.tile(clszf_center, [1, Ns, 1, 1, 1])
        clszf_center = clszf_center.flatten(0, 1)

        N,C,H,W = clsxf_sig.size()
        clsxf_sig = clsxf_sig.view((Nbatch,Ns,C,H,W))
        clsxf_sig = torch.unsqueeze(clsxf_sig, dim=1)
        clsxf_sig = torch.tile(clsxf_sig, [1, Nt, 1, 1, 1,1])
        clsxf_sig = clsxf_sig.flatten(0, 2)
        sim = torch.abs(clszf_center-clsxf_sig).sum(dim=1,keepdim=True).sigmoid()
        fpnxf = self.fpn(xfbb[:2])
        N, C, H, W = fpnxf.size()
        fpnxf = fpnxf.view((Nbatch, Ns, C, H, W))
        fpnxf = torch.unsqueeze(fpnxf, dim=1)
        fpnxf = torch.tile(fpnxf, [1, Nt, 1, 1, 1, 1])
        fpnxf = fpnxf.flatten(0, 2)

        cls_corr,loc_corr = self.head(fpnxf,sim)
        #
        cls_corr = self.log_softmax(cls_corr)
        cls_loss_corr = select_cross_entropy_loss(cls_corr, label_cls_corr, True, rank=rank)
        loc_loss_corr = select_iou_loss(loc_corr, delta_corr, label_cls_corr, rank=rank)




        outputs = {}
        outputs['total_loss'] = cls_loss+cls_loss_corr+loc_loss_corr
        outputs['cls_loss_corr'] = cls_loss_corr
        outputs['loc_loss_corr'] = loc_loss_corr
        outputs['cls_loss'] = cls_loss


        return outputs

    def compute_locations_per_level(self, feat, stride,im_c):
        h,w = feat.size()[2:]
        ori = im_c - w // 2 * stride

        device = feat.device
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + ori

        return [locations,]

if __name__=='__main__':
    from siamban.core.config import cfg
    from siamban.models.backbone.mobileone_strideS16OutTwo import reparameterize_model_train, \
        reparameterize_model_allskipscale

    cfg.BACKBONE.TYPE = "mobileones16outtwo"
    cfg.BAN.channels = 256
    model = ModelBuilder(cfg)
    model.backbone = reparameterize_model_allskipscale(model.backbone)
    model = model.cuda()


    data = {}
    data['template'] = torch.randn((2,3,3,160,160)).cuda()
    data['search'] = torch.randn((2,6,3,224,224)).cuda()

    out= model(data)