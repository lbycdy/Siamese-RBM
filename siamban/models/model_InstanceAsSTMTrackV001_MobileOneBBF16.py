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

class Adjustayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Adjustayer, self).__init__()
        self.adjustor = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=True,padding=1)

        nn.init.orthogonal_(self.adjustor.weight.data)
        nn.init.zeros_(self.adjustor.bias.data)
    def forward(self, feature):
        return self.adjustor(feature)
class RPNHead(nn.Module):
    def __init__(self,cin = 256,feathidden=256):
        super(RPNHead, self).__init__()
        self.head1 = nn.Sequential(
            nn.Conv2d(cin, feathidden, kernel_size=3, bias=True, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feathidden, feathidden, kernel_size=3, bias=True, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feathidden, feathidden, kernel_size=3, bias=True, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(feathidden, feathidden, kernel_size=3, bias=True, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feathidden, feathidden, kernel_size=3, bias=True, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feathidden, feathidden, kernel_size=3, bias=True, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feathidden, 1, kernel_size=3, bias=True, padding=1),

        )
        self.loc = nn.Sequential(
            nn.Conv2d(feathidden, feathidden, kernel_size=3, bias=True, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feathidden, feathidden, kernel_size=3, bias=True, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feathidden, feathidden, kernel_size=3, bias=True, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feathidden, 4, kernel_size=3, bias=True, padding=1),

        )

        self.loc_scale = nn.Parameter(torch.ones(1))

    def forward(self, feat):
        head1 = self.head1(feat)
        cls = self.cls(head1)
        loc = self.loc(head1)
        return cls,torch.exp(loc*self.loc_scale[0])

class ModelBuilder(nn.Module):
    def __init__(self,cfg=None):
        super(ModelBuilder, self).__init__()
        outch = cfg.BAN.channels
        self.num_feat_pertemplate = cfg.TRAIN.NUM_FEAT_PERTEMPLATE
        # build backbone
        self.backbone = mobileone(variant='s0')
        # build adjust layer
        self.neck = Adjustayer(in_channels=256,out_channels=outch)
        # build rpn head
        self.head = RPNHead(cin=outch+self.num_feat_pertemplate*3,feathidden=outch)
        self.cnt = 0

    def getfeat(self,feat,box):
        _,C,feath,featw = feat.size()
        xmin,ymin,xmax,ymax = box
        bw = xmax - xmin
        bh = ymax - ymin
        n = bw * bh
        if n > self.num_feat_pertemplate:
            xc = (xmin + xmax) / 2.0
            yc = (ymin + ymax) / 2.0
            if bw >= bh:
                bw = int(math.floor(float(self.num_feat_pertemplate) / bh))
                xmin = xc - bw / 2
                xmax = xc + bw / 2
                xmin, xmax = map(int, [xmin, xmax])
                xmin = max(0, xmin)
                xmax = min(feath, xmax)
            else:
                bh = int(math.floor(float(self.num_feat_pertemplate) / bw))
                ymin = yc - bh / 2
                ymax = yc + bh / 2
                ymin, ymax = map(int, [ymin, ymax])
                ymin = max(0, ymin)
                ymax = min(feath, ymax)
        f = feat[0, :,ymin:ymax, xmin:xmax].flatten(1, 2)
        return f
    def template(self, z,boxonfeat):
        z -= 114.0
        z /= 58.0
        zf = self.backbone(z)
        zf = self.neck(zf[1])
        feat = self.getfeat(zf,boxonfeat)
        for _ in range(20):
            n = feat.size()[1]
            if n == self.num_feat_pertemplate:
                break
            gap = min(self.num_feat_pertemplate - n, n)
            feat = torch.cat([feat, feat[:, :gap]], dim=1)
        feat = torch.tile(feat,(1,3))
        self.kernel = feat.transpose(0,1).unsqueeze(2).unsqueeze(3)

    def track(self, x):
        # import numpy as np
        # x = np.load("/home/ethan/SGS_IPU_SDK_v1.1.6/x_crop.npy").transpose((0, 3, 1, 2))
        # x = torch.from_numpy(x).cuda()
        # print(x.size(),"xxxx")
        imgsize = x.size()[2]
        x -= 114.0
        x /= 58.0
        xf = self.backbone(x)
        xf = self.neck(xf[1])
        featsim = F.conv2d(xf, self.kernel)
        featsim = F.softmax(featsim, dim=1)
        feat = torch.cat([xf, featsim], dim=1)
        cls, loc = self.head(feat)
        s = (float(imgsize) / 10.0)
        loc *= s
        return cls,loc


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
        template = data['template'].float().to(rank, non_blocking=True).flatten(0,1)
        search = data['search'].float().to(rank, non_blocking=True)
        label_cls = data['label_cls'].float().to(rank, non_blocking=True)
        label_loc = data['label_loc'].float().to(rank, non_blocking=True)
        template_box_onfeat = data['template_box_onfeat'].data.cpu().flatten(0,1).numpy()
        imgsize = search.size()[2]
        label_loc /= (float(imgsize) / 10.0)

        template -= 114.0
        template /= 58.0
        search -= 114.0
        search /= 58.0
        # get feature
        zf = self.backbone(template)[1]
        xf = self.backbone(search)[1]
        zf = self.neck(zf)
        xf = self.neck(xf)
        feath = zf.size()[2]
        kernel = []
        for ib in range(Nbatch):
            k = []
            for j in range(Nt):
                idx = ib*Nt+j
                # xmin,ymin,xmax,ymax = template_box_onfeat[j]
                xmin,ymin,xmax,ymax = template_box_onfeat[idx]#codemodified on 20230815

                feat = zf[idx]
                bw = xmax - xmin
                bh = ymax - ymin
                n = bw*bh
                if n>self.num_feat_pertemplate:
                    xc = (xmin+xmax)/2.0
                    yc = (ymin + ymax)/2.0
                    if bw>=bh:
                        bw = int(math.floor(float(self.num_feat_pertemplate)/bh))
                        xmin = xc - bw/2
                        xmax = xc + bw/2
                        xmin,xmax = map(int,[xmin,xmax])
                        xmin = max(0,xmin)
                        xmax = min(feath,xmax)
                    else:
                        bh = int(math.floor(float(self.num_feat_pertemplate) / bw))
                        ymin = yc - bh / 2
                        ymax = yc + bh / 2
                        ymin, ymax = map(int, [ymin, ymax])
                        ymin = max(0, ymin)
                        ymax = min(feath, ymax)
                f = feat[:,ymin:ymax,xmin:xmax].flatten(1,2)
                for _ in range(20):
                    n = f.size()[1]
                    if n==self.num_feat_pertemplate:
                        break
                    gap = min(self.num_feat_pertemplate-n,n)
                    f = torch.cat([f,f[:,:gap]],dim=1)
                k.append(f)
            k = torch.cat(k,dim=1)
            kernel.append(k)
        kernel = torch.stack((kernel)).transpose(1,2).unsqueeze(3).unsqueeze(4)
        kernel = kernel.flatten(0,1)
        _, C, H, W = xf.size()
        featsim = xf.view(1, -1, H, W)
        featsim = F.conv2d(featsim, kernel, groups=Nbatch).view(Nbatch,-1,H,W)
        featsim = F.softmax(featsim,dim=1)

        feat = torch.cat([xf,featsim],dim=1)
        cls_pred,loc_pred = self.head(feat)

        loc_loss = select_iou_loss(loc_pred, label_loc, label_cls, rank=rank)

        cls_pred = cls_pred.reshape(-1,)
        label_cls = label_cls.reshape(-1,)

        pos = label_cls.data.ge(0).nonzero().squeeze().to(rank)
        # flag = label_cls==1
        # print(label_cls.shape,len(pos),flag.sum())

        cls_pred = torch.index_select(cls_pred, 0, pos)
        label_cls = torch.index_select(label_cls, 0, pos).float()
        total_num_pos = reduce_sum(label_cls.new_tensor([label_cls.numel()]), numgpus).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(numgpus), 1.0)
        if cfg.TRAIN.FLAG_FOCAL_LOSS:
            cls_loss = tvops.sigmoid_focal_loss(cls_pred, label_cls, alpha=cfg.TRAIN.FOCAL_ALPHA, gamma=cfg.TRAIN.FOCAL_GAMA,reduction='sum')/num_pos_avg_per_gpu
        else:
            cls_loss = F.binary_cross_entropy_with_logits(cls_pred, label_cls, reduction="sum")/num_pos_avg_per_gpu
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT*cls_loss+cfg.TRAIN.LOC_WEIGHT*loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
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
    from siamban.models.backbone.mobileone_stride import reparameterize_model_allskipscale

    model = ModelBuilder(cfg)
    model.backbone = reparameterize_model_allskipscale(model.backbone)
    model = model.cuda()

    data = {}
    data['template'] = torch.randn((2,3,3,256,256)).cuda()
    data['search'] = torch.randn((2,3,256,256)).cuda()
    label_cls = torch.randn((2, 1, 31, 31))
    label_cls = torch.where(label_cls>0.5,1,0)
    data['label_cls'] = label_cls.cuda()
    data['label_loc'] = torch.randn((2,4,31,31)).cuda()
    boxes = [[7,7,22,22],[7,7,22,22],[7,7,22,22],[7,7,22,22],[7,7,22,22],[7,7,22,22]]
    boxes =torch.tensor(boxes).view((2,3,-1)).cuda()
    data['template_box_onfeat'] = boxes

    #
    out= model(data)