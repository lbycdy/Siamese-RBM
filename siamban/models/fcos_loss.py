import torch

from torch import nn
import os
# from siamban.models.iou_loss import IOULoss
from torch.nn import functional as F

import torchvision.ops as tvops

INF = 100000000




class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super(IOULoss, self).__init__()
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

        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg, world_size):
        self.world_size = world_size
        # self.cls_loss_func = focal_loss(
        #     alpha= cfg.TRAIN.FCOS.LOSS_ALPHA,
        #     gamma=cfg.TRAIN.FCOS.LOSS_GAMMA
        # )

        # 各层特征相对于输入图像的下采样倍数 [8, 16, 32, 64, 128]
        self.fpn_strides = cfg.TRAIN.FCOS.FPN_STRIDES
        # 中心采样半径，通常是1.5 它会乘以各层的下采样步长
        self.center_sampling_radius = cfg.TRAIN.FCOS.CENTER_SAMPLING_RADIUS
        # IoU Loss 类型："iou", "linear_iou" or "giou"，本文选giou
        self.iou_loss_type = cfg.TRAIN.FCOS.IOU_LOSS_TYPE
        # 是否使用normalizing regression targets策略
        # (会对回归标签使用下采样步长进行归一化)
        # 本文为True
        self.norm_reg_targets = cfg.TRAIN.FCOS.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        # self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")

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
        # 每层特征图的点的数量
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)

        # 论文中分配box策略
        # 根据预设比例对不同level特征图分配不同的大小的box
        # labels：5层特征图每个point对应的label size：[torch.Size([21486])]
        # reg_targets：5层特征图每个point对应的box的四个坐标 size：[torch.Size([21486, 4])]
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets
        )
        # batch_size为len(labels)
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            # 将label[0]切分为了[torch.Size([16128]), torch.Size([4032]), torch.Size([1008]), torch.Size([252]), torch.Size([66])]
            # 对应这5个特征图
            # reg_targets同理
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        # 将多个图片的同一level的label和reg_targets合并成一个tensor
        for level in range(len(points)):
            # 对一个batch内的所有图片的所有point在行的维度上进行concat
            # 后追加到labels_level_first
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            # 对一个batch内的所有图片的所有point在行的维度上进行concat
            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im[:, :4]
            area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            labels_per_im = targets[im_i][:, 4]
            # area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

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


            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        # (前景)类别数量
        num_classes = box_cls[0].size(1)

        # 1. 标签分配

        # 两个list，len=num_levels，
        # list中的每项是对应特征层所有图片的特征点的类别标签和回归标签 shape分别为:
        # (num_images*num_points_level_l,) (num_images*num_points_this_level_l,4)
        labels, reg_targets = self.prepare_targets(locations, targets)

        # 2. 排列和变换各层预测结果和标签的维度，将所有特征层的结果拼接在一起，以便后续计算

        # 以下这批list的长度都等于特征层数量 len=num_levels

        box_cls_flatten = []
        box_regression_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        # 2.1 permute & flatten
        # 依次处理各个特征层
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))

        # 2.2 concat
        # 将所有特征层(所有图片)的预测结果拼接在一起
        # (num_points_all_levels_batches,num_classes)
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        # (num_points_all_levels_batches,4)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        # 将所有特征层(所有图片)的标签拼接在一起
        # (num_points_all_levels_batches,)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        # (num_points_all_levels_batches,4)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        # 3. 获取正样本的回归预测和centerness预测(因为回归损失和centerness损失仅对正样本计算)

        # (num_pos,) 正样本(特征点)索引
        # torch.nonzero(labels_flatten > 0)返回的shape是(num_points_all_levels_batches,1)
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        # (num_pos,4) 正样本对应的回归预测
        box_regression_flatten = box_regression_flatten[pos_inds]
        # (num_pos,4)
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        # 4. 计算分类损失(正负样本都要计算)

        # 在所有GPU上进行同步，使得每个GPU得到相同的正样本数量，是一个同步操作
        # num_gpus = get_num_gpus()
        num_gpus = self.world_size
        # sync num_pos from all gpus
        # 所有gpu上正样本数量的总和
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        # 所有gpu上正样本数量的均值
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        # 计算分类损失 多分类Focal Loss

        labels_flatten = F.one_hot(labels_flatten.to(torch.int64), num_classes=2)
        labels_flatten = labels_flatten[:, 1:]
        cls_loss = tvops.sigmoid_focal_loss(box_cls_flatten, labels_flatten.float(),
                                            reduction='sum') / num_pos_avg_per_gpu
        # 若该批次中有正样本，则进一步计算回归与centerness损失
        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss

            # 计算所有正样本centerness标签的总和(每个gpu都求和然后计算均值) 用于对回归损失进行归一化
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)
            # 使用centerness标签对loss加权，离物体中心越近的权重越大
            # 最后再除以所有正样本centerness标签的总和
            reg_loss = self.box_reg_loss_func(
                # (num_pos,4)
                box_regression_flatten,
                # (num_pos,4)
                reg_targets_flatten,
                # (num_pos,)
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu

        # 若该批次中没有正样本，则回归设置为0
        else:
            # box_regression_flatten的shape是(0,4)
            reg_loss = box_regression_flatten.sum()
        return cls_loss, reg_loss


def make_fcos_loss_evaluator(cfg,world_size):
    loss_evaluator = FCOSLossComputation(cfg,world_size)
    return loss_evaluator



if __name__=='__main__':
    from siamban.core.config import cfg
    loss = FCOSLossComputation(cfg)
    fpn_strides = [8, 16, 32]
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
    features = [torch.randn((28, 128, 28, 52)).cuda(), torch.randn((28, 128, 14, 26)).cuda(),
                torch.randn((28, 128, 7, 13)).cuda()]
    box_cls = [torch.randn((1, 1, 28, 52)).cuda(), torch.randn((1, 1, 14, 26)).cuda(),
               torch.randn((1, 1, 7, 13)).cuda()]
    box_regression = [torch.randn((1, 8, 28, 52)).cuda(), torch.randn((1, 8, 14, 26)).cuda(),
                      torch.randn((1, 8, 7, 13)).cuda()]
    # points = [[5, 5, 80, 10, 80, 100, 5, 120, 8000], [90, 90, 150, 100, 160, 150, 80, 150, 5000]]
    points = [[20,20,30,30,1],[25,25,30,30,1],[25,24,40,40,1]]
    targets = [torch.tensor(points).cuda(), ]
    locations = compute_locations(features)
    for l in locations:
        print(l.size())

    lss = loss(locations, box_cls, box_regression, targets)
    print("-------------------")
    for l in lss:
        print(l)