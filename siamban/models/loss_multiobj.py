"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn
import os
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable


INF = 100000000
import torchvision.ops as tvops
from torchvision.ops import sigmoid_focal_loss

class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)

        gious = ious - (ac_uion - area_union) / ac_uion

        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()



def reduce_sum(tensor,num_gpus):
    if num_gpus <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class FCOSAlikeLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.flag_focal_loss = cfg.TRAIN.FLAG_FOCAL_LOSS
        self.norm_reg_targets = cfg.TRAIN.NORM_REG_TARGETS
        self.fpn_strides = cfg.TRAIN.FPN_STRIDES
        self.center_sampling_radius = cfg.TRAIN.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.TRAIN.IOU_LOSS_TYPE
        self.focal_alpha = cfg.TRAIN.FOCAL_ALPHA
        self.focal_gama = cfg.TRAIN.FOCAL_GAMA
        object_sizes_of_interest = cfg.TRAIN.OBJECT_SIZES_OF_INTEREST
        assert len(object_sizes_of_interest)-1==len(self.fpn_strides)
        self.object_sizes_of_interest = []
        for i in range(len(object_sizes_of_interest)-1):
            self.object_sizes_of_interest.append([object_sizes_of_interest[i],object_sizes_of_interest[i+1]])


        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(self.object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targeti = targets[im_i]
            if len(targeti)>0:
                bboxes = targeti[:, :4]
                labels_per_im = targeti[:, 4]
                area = targeti[:, 5]

                l = xs[:, None] - bboxes[:, 0][None]
                t = ys[:, None] - bboxes[:, 1][None]
                r = bboxes[:, 2][None] - xs[:, None]
                b = bboxes[:, 3][None] - ys[:, None]
                reg_targets_per_im = torch.stack([l, t, r, b], dim=2)#torch.Size([2880, 1, 4])

                if self.center_sampling_radius > 0:
                    is_in_boxes = self.get_sample_region(
                        bboxes,
                        self.fpn_strides,
                        self.num_points_per_level,
                        xs, ys,
                        radius=self.center_sampling_radius
                    )
                else:
                    # no center sampling, it will use all the locations within a ground-truth box
                    is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

                max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
                # limit the regression range for each location
                is_cared_in_the_level = \
                    (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                    (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

                locations_to_gt_area = area[None].repeat(len(locations), 1)
                locations_to_gt_area[is_in_boxes == 0] = INF
                locations_to_gt_area[is_cared_in_the_level == 0] = INF

                # if there are still more than one objects for a location,
                # we choose the one with minimal area
                locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

                reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
                labels_per_im = labels_per_im[locations_to_gt_inds]
                labels_per_im[locations_to_min_area == INF] = 0
            else:
                npos = len(xs)
                device = xs.device
                labels_per_im = torch.zeros((npos,)).to(device)
                reg_targets_per_im = torch.zeros((npos,4)).to(device)


            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, targets,num_gpus,centerness=None):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        # print(num_classes,":num_classes",box_cls[0].size())
        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
        if centerness is not  None:
            centerness_flatten = []
            for l in range(len(labels)):
                centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        if centerness is not None:
            centerness_flatten = torch.cat(centerness_flatten, dim=0)
            centerness_flatten = centerness_flatten[pos_inds]

        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()]),num_gpus).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        # cls_loss = self.cls_loss_func(
        #     box_cls_flatten,
        #     labels_flatten.int()
        # ) / num_pos_avg_per_gpu

        label_c = []
        for i in range(num_classes):
            l = torch.eq(labels_flatten,i+1)
            label_c.append(l.float())
        label_c = torch.stack(label_c,dim=1)
        # r = label_c.sum()/label_c.numel()
        # print(label_c.sum(),label_c.numel())
        # print(box_cls_flatten.size(),label_c.size())
        if self.flag_focal_loss:
            cls_loss = sigmoid_focal_loss(box_cls_flatten, label_c, alpha=self.focal_alpha, gamma=self.focal_gama,reduction='sum') /num_pos_avg_per_gpu
        else:
            cls_loss = F.binary_cross_entropy_with_logits(box_cls_flatten,label_c,reduction='sum')/num_pos_avg_per_gpu


        # cls_loss = F.binary_cross_entropy_with_logits(box_cls_flatten, label_c) /num_pos_avg_per_gpu
        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum(),num_gpus).item() / float(num_gpus)
            if centerness is not None:
                reg_loss = self.box_reg_loss_func(
                    box_regression_flatten,
                    reg_targets_flatten,
                    centerness_targets
                ) / sum_centerness_targets_avg_per_gpu
                centerness_loss = self.centerness_loss_func(
                    centerness_flatten,
                    centerness_targets
                ) / num_pos_avg_per_gpu
            else:
                reg_loss = self.box_reg_loss_func(
                    box_regression_flatten,
                    reg_targets_flatten,
                    None
                ) / sum_centerness_targets_avg_per_gpu
                centerness_loss = None

        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]),num_gpus)
            if centerness is not None:
                centerness_loss = centerness_flatten.sum()
            else:
                centerness_loss = None

        return cls_loss, reg_loss, centerness_loss,labels



class FCOSAlikeLossComputation_TargetIn(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.flag_focal_loss = cfg.TRAIN.FLAG_FOCAL_LOSS
        self.iou_loss_type = cfg.TRAIN.IOU_LOSS_TYPE
        self.focal_alpha = cfg.TRAIN.FOCAL_ALPHA
        self.focal_gama = cfg.TRAIN.FOCAL_GAMA
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)

    def __call__(self, labels,reg_targets, box_cls, box_regression,num_gpus):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls.size(0)
        num_classes = box_cls.size(1)
        box_cls_flatten = box_cls.permute(0, 2, 3, 1).reshape(-1)
        box_regression_flatten = box_regression.permute(0, 2, 3, 1).reshape(-1, 4)
        labels_flatten = labels.reshape(-1)
        reg_targets_flatten = reg_targets.reshape(-1, 4)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        # print(box_regression_flatten.size(),reg_targets_flatten.size(),":before")
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        # print(box_regression_flatten.size(),reg_targets_flatten.size(),":after")

        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()]),num_gpus).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        if self.flag_focal_loss:
            cls_loss = sigmoid_focal_loss(box_cls_flatten, labels_flatten, alpha=self.focal_alpha, gamma=self.focal_gama,reduction='sum') /num_pos_avg_per_gpu
        else:
            cls_loss = F.binary_cross_entropy_with_logits(box_cls_flatten,labels_flatten,reduction='sum')/num_pos_avg_per_gpu


        # cls_loss = F.binary_cross_entropy_with_logits(box_cls_flatten, label_c) /num_pos_avg_per_gpu
        if pos_inds.numel() > 0:
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                None
            ) / num_pos_avg_per_gpu

        else:
            reg_loss = box_regression_flatten.sum()

        return cls_loss, reg_loss

if __name__=='__main__':
    from config.config import cfg
    import numpy as np
    import cv2
    print(cfg)

    def checkvector_right(pa, pb, location):
        xa, ya = pa[:, 0], pa[:, 1]
        xb, yb = pb[:, 0], pb[:, 1]
        vabx = xb - xa
        vaby = yb - ya
        vamx = location[:, [0]] - xa[None,]
        vamy = location[:, [1]] - ya[None,]
        V = vabx * vamy - vaby * vamx
        return V >= 0

    loss = FCOSAlikeLossComputation(cfg)
    fpn_strides = [8,16,32]


    def compute_locations(features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations_per_level(
                h, w, fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations


    def compute_locations_per_level(h, w, stride, device):
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
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2

        return locations

    features = [torch.randn((1,128,28,52)).cuda(),torch.randn((1,128,14,26)).cuda(),torch.randn((1,128,7,13)).cuda()]
    box_cls = [torch.randn((1,2,28,52)).cuda(),torch.randn((1,2,14,26)).cuda(),torch.randn((1,2,7,13)).cuda()]
    box_regression = [torch.randn((1,4,28,52)).cuda(),torch.randn((1,4,14,26)).cuda(),torch.randn((1,4,7,13)).cuda()]
    centerness = [torch.randn((1,1,28,52)).cuda(),torch.randn((1,1,14,26)).cuda(),torch.randn((1,1,7,13)).cuda()]

    # features = [torch.randn((1, 128, 7, 13)).cuda()]
    # box_cls = [  torch.randn((1, 1, 7, 13)).cuda()]
    # box_regression = [torch.randn((1, 8, 7, 13)).cuda()]


    for xshift in range(0,416,5):
        for yshift in range(0,224,5):
                points = [[5+xshift, 5+yshift, 400+xshift, 120+yshift,1], [90+xshift, 90+yshift, 150+xshift, 150+yshift,2]]
                for i in range(len(points)):
                    xmin,ymin,xmax,ymax = points[i][:4]
                    print(points[i])
                    a = (xmax-xmin)*(ymax-ymin)
                    points[i].append(a)
                    print(a)
                # points = [[90+xshift, 90+yshift, 150+xshift, 100+yshift, 160+xshift, 150+yshift, 80+xshift, 150+yshift,5000]]
                img = np.zeros((224, 416, 3), dtype=np.uint8)

                for p in points:
                    x1, y1, x2, y2, cid,a = map(int, p)
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255))
                cv2.namedWindow("img",cv2.NORM_MINMAX)
                cv2.imshow("img",img)
                netinh,netinw = 224,416
                targets = [torch.tensor(points).cuda().float(),]


                locations = compute_locations(features)

                cls,loc,cent,labels = loss(locations, box_cls, box_regression, centerness,targets,num_gpus=1)
                for i in range(len(labels)):
                    print(labels[i].size(),box_cls[i].size())
                exit()
                hws = [[28, 52], [14, 26], [7, 13], [4, 7], [2, 4]]
                for idx, l in enumerate(labels):
                    print(torch.unique(l),idx)

                    h, w = hws[idx]
                    limg = l.reshape(h, w).data.cpu().numpy()
                    limg *= 255
                    limg = limg.astype(np.uint8)
                    head1 = cv2.resize(limg, (netinw, netinh))
                    heat_img = cv2.applyColorMap(head1, cv2.COLORMAP_JET)
                    add_img = cv2.addWeighted(img, 0.5, heat_img, 0.5, 0)
                    cv2.namedWindow("%d" % idx, cv2.NORM_MINMAX)
                    cv2.imshow("%d" % idx, add_img)
                key = cv2.waitKey()
                if key==27:
                    exit()
