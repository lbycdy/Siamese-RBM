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
        self.ca1 = ChannelAttention(out_channels[0], 128)
        self.ca2 = ChannelAttention(out_channels[1], 128)
    def forward(self, features,flagcentercrop=False):
        downsample1 = self.downsample1(features[0])
        downsample2 = self.downsample2(features[1])
        downsample1 = self.ca1(downsample1) * downsample1
        downsample2 = self.ca2(downsample2) * downsample2
        out = downsample1+downsample2
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
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False,padding=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )

        self.xorr_search = nn.Conv2d(hidden, hidden, kernel_size=5, bias=False,padding=2)
        self.xorr_kernel = nn.Conv2d(hidden, hidden, kernel_size=5, bias=True)
        self.xorr_activate = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, bias=True, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, search,kernel):
        # print(search.size(),kernel.size(),":searchkernel")

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
        feature_in = 64
        self.head1 = nn.Sequential(
            nn.Conv2d(cin*2, feature_in*2, kernel_size=1, bias=False, padding=0),
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
            nn.Conv2d(feature_in, 2, kernel_size=3, bias=True, padding=1),
        )
        self.loc = nn.Sequential(
            nn.Conv2d(feature_in, feature_in, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(feature_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in, feature_in, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(feature_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_in, 4, kernel_size=3, bias=True, padding=1),
        )

        self.loc_scale = nn.Parameter(torch.ones(1))

    def forward(self, xf,zf):
        depthcorr = self.depthcorr(xf,zf)
        head1 = self.head1(torch.cat([depthcorr,xf],dim=1))
        cls = self.cls(head1)
        loc = self.loc(head1)
        return cls,torch.exp(loc*self.loc_scale[0])

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
        self.neck = AdjustAllLayerFPNAddFlagCenterCrop(in_channels=[256,1024],out_channels=[outch,outch],center_size=7)
        # build rpn head
        self.head = RPNHead(cin=outch)
        self.dethead = DetHead(outch,outch,cfg.TRAIN.NUM_CLASSES)
        self.cls_temp = nn.Linear(outch, cfg.TRAIN.NUM_CLASSES)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.softmaxloss = nn.CrossEntropyLoss()
        self.cnt = 0



    def template(self, z):
        z -= 114.0
        z /= 58.0
        zf = self.backbone(z)
        zf = self.neck(zf[1:], flagcentercrop=True)
        self.zf = zf
        zf_ave = self.avg_pool(zf)
        b = zf_ave.size()[0]
        zf_cls = zf_ave.view(b, -1)
        cls_temp = self.cls_temp(zf_cls)
        print(cls_temp.size())
        cls_temp = cls_temp.softmax(1)
        print(cls_temp.min(),cls_temp.max(),cls_temp.sum())
        idx = torch.argsort(cls_temp,1,descending=True)[0][:10]
        print(idx,cls_temp[0,idx],":cls_temp")

    def track(self, x):
        # import numpy as np
        # x = np.load("/home/ethan/SGS_IPU_SDK_v1.1.6/x_crop.npy").transpose((0, 3, 1, 2))
        # x = torch.from_numpy(x).cuda()
        # print(x.size(),"xxxx")
        imgsize = x.size()[2]
        x -= 114.0
        x /= 58.0
        xf = self.backbone(x)
        xf = self.neck(xf[1:])
        cls, loc = self.head(xf, self.zf)
        cls_det,loc_det = self.dethead(xf)
        cls_detsigmoid = cls_det.sigmoid()
        print(cls_detsigmoid.max(),cls_detsigmoid.min(),":cls_detsigmoid")
        # loc_det *= 8.0
        location_search = self.compute_locations_per_level(xf,8,imgsize//2)[0]
        boxlists = self.forward_for_single_feature_map(location_search, cls_det, loc_det, None, 0.02)
        # print(boxlists)
        # exit()
        # boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        # for i in range(239):
        #     print(i,cls_det[0,i].min(),cls_det[0,i].max())


        # loc *= 22.4
        # loc *= 8.0
        # exit()
        return cls,loc,boxlists[0]

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            scorethresh):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        self.pre_nms_thresh = scorethresh
        self.pre_nms_top_n = 5
        self.min_size = 2
        self.nms_thresh = 0.35
        self.fpn_post_nms_top_n = 100
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)


        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.contiguous().view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        if centerness is not None:
            centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
            centerness = centerness.reshape(N, -1).sigmoid()
            box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)
            h, w = 224,224
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            xywh_boxes = boxlist.convert("xywh").bbox
            _, _, ws, hs = xywh_boxes.unbind(dim=1)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

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
            template_cid = data['template_cid'].cuda().float().flatten(0,1)
            label_cls_det = data['cls_det'].cuda().float().flatten(0,1)
            delta_det = data['delta_det'].cuda().float().flatten(0,1)
            label_cls_corr = data['cls_corr'].cuda().float().flatten(0,1)
            delta_corr = data['delta_corr'].cuda().float().flatten(0,1)

        else:
            template = data['template'].float().to(rank, non_blocking=True).flatten(0,1)
            search = data['search'].float().to(rank, non_blocking=True).flatten(0,1)
            template_cid = data['template_cid'].float().to(rank, non_blocking=True).flatten(0,1)
            label_cls_det = data['cls_det'].float().to(rank, non_blocking=True).flatten(0,1)
            delta_det = data['delta_det'].float().to(rank, non_blocking=True).flatten(0,1)
            label_cls_corr = data['cls_corr'].float().to(rank, non_blocking=True).flatten(0,1)
            delta_corr = data['delta_corr'].float().to(rank, non_blocking=True).flatten(0,1)


        # print(template.size(),search.size(),label_cls.size(),label_loc.size(),label_target.size())
        # print(torch.unique(label_cls))
        # print(template.size(),search.size(),":tempsearch")
        imgsize = search.size()[2]
        delta_det /= (float(imgsize) / 10.0)
        delta_corr /= (float(imgsize) / 10.0)

        template -= 114.0
        template /= 58.0
        search -= 114.0
        search /= 58.0
        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        zf = self.neck(zf[1:], flagcentercrop=True)
        xf = self.neck(xf[1:])


        zf_ave = self.avg_pool(zf)
        b = zf_ave.size()[0]
        zf_cls = zf_ave.view(b, -1)
        cls_temp = self.cls_temp(zf_cls)
        temploss_cls = self.softmaxloss(cls_temp, template_cid.long())
        # print(cls_loss_temp,"cls_loss_temp")
        cls_det,loc_det = self.dethead(xf)

        zf = torch.unsqueeze(zf, dim=1)
        zf = torch.tile(zf, [1, Ns, 1, 1, 1])
        zf = zf.flatten(0, 1)

        N,C,H,W = xf.size()
        xf = xf.view((Nbatch,Ns,C,H,W))
        xf = torch.unsqueeze(xf, dim=1)
        xf = torch.tile(xf, [1, Nt, 1, 1, 1,1])
        xf = xf.flatten(0, 2)
        cls_corr,loc_corr = self.head(xf,zf)
        cls_corr = self.log_softmax(cls_corr)
        cls_loss_corr = select_cross_entropy_loss(cls_corr, label_cls_corr, True, rank=rank)
        loc_loss_corr = select_iou_loss(loc_corr, delta_corr, label_cls_corr, rank=rank)

        cls_det = cls_det.permute(0, 2, 3, 1).reshape(-1, cfg.TRAIN.NUM_CLASSES)
        label_cls_det = label_cls_det.reshape(-1)
        """
        modified after 20230614(inclusive)
        pos = label_cls_det.data.ge(0).nonzero().squeeze().to(rank)
        pred = torch.index_select(cls_det, 0, pos)
        label = torch.index_select(label_cls_det, 0, pos).long()
        labelbce = F.one_hot(label,num_classes=cfg.TRAIN.NUM_CLASSES).float()
        """
        indexsel = label_cls_det.data.ge(0).nonzero().squeeze().to(rank)
        pred = torch.index_select(cls_det, 0, indexsel)
        label = torch.index_select(label_cls_det, 0, indexsel).long()
        labelbce = F.one_hot(label, num_classes=cfg.TRAIN.NUM_CLASSES+1).float()[:,1:]

        if cfg.TRAIN.FLAG_FOCAL_LOSS:
            pos_inds = torch.nonzero(label > 0).squeeze(1)
            total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()]),numgpus).item()
            num_pos_avg_per_gpu = max(total_num_pos / float(numgpus), 1.0)
            cls_loss_det = tvops.sigmoid_focal_loss(pred, labelbce, alpha=cfg.TRAIN.FOCAL_ALPHA, gamma=cfg.TRAIN.FOCAL_GAMA,reduction='sum')/num_pos_avg_per_gpu

        else:
            cls_loss_det = F.binary_cross_entropy_with_logits(pred, labelbce, reduction='mean')
        loc_loss_det = select_iou_loss_multiclass(loc_det, delta_det, label_cls_det, rank=rank)


        outputs = {}
        outputs['total_loss'] = temploss_cls+cls_loss_corr+loc_loss_corr + cls_loss_det+loc_loss_det
        outputs['temploss_cls'] = temploss_cls
        outputs['cls_loss_corr'] = cls_loss_corr
        outputs['loc_loss_corr'] = loc_loss_corr
        outputs['cls_loss_det'] = cls_loss_det
        outputs['loc_loss_det'] = loc_loss_det


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
    model = ModelBuilder(cfg)
    model.backbone = reparameterize_model_allskipscale(model.backbone)
    model = model.cuda()
    search = torch.randn((4,3,224,224)).cuda()
    out = model.backbone(search)
    for o in out:
        print(o.size())
    exit()


    data = {}
    data['template'] = torch.randn((1,2,3,160,160)).cuda()
    data['search'] = torch.randn((1,4,3,224,224)).cuda()
    data['template_cid'] = torch.tensor([0,2]).cuda().view(1,-1)
    searchboxes = [[0,0,35,36,0],[56,46,129,211,1],[60,80,200,150,1],[55,89,168,180,199],]
    for i in range(len(searchboxes)):
        xmin,ymin,xmax,ymax,cid = searchboxes[i]
        w = xmax - xmin
        h = ymax - ymin
        searchboxes[i].append(w*h)
    searchboxes_corr = searchboxes[:]
    for i in range(len(searchboxes_corr)):
        searchboxes_corr[i][4]=1

    searchboxes = [torch.tensor(box).cuda().view(1,-1).float() for box in searchboxes]
    searchboxes_corr = [torch.tensor(box).cuda().view(1,-1).float() for box in searchboxes_corr]
    searchboxes_corr.extend(searchboxes_corr)
    print(len(searchboxes_corr))
    data['searchboxes'] =searchboxes
    data['searchboxes_corr'] =searchboxes_corr

    #
    out= model(data)