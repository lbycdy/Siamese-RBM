from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from siamban.core.config import cfg
from siamban.tracker.base_tracker import SiameseTracker
from siamban.utils.bbox import corner2center

import cv2

class SiamBANTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamBANTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)
        self.model = model
        self.model.eval()

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):
        # print(point.shape,":point")

        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):

        if cfg.TRAIN.FLAG_SIGMOID_LOSS:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.sigmoid().detach()[:, 1].cpu().numpy()

        else:
            if self.cls_out_channels == 1:
                score = score.permute(1, 2, 3, 0).contiguous().view(-1)
                score = score.sigmoid().detach().cpu().numpy()
            else:
                score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
                score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.array([117.0,117.0,117.0])
        # print(self.center_pos, img.mean())

        # get crop
        z_crop,_ = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        if cfg.TRAIN.DATA_NORMALIZE:
            if cfg.TRAIN.DATA_NORMALIZE_MODE == 0:
                z_crop /= 255.0
                z_crop -= self.mean
                z_crop /= self.std
            elif cfg.TRAIN.DATA_NORMALIZE_MODE == 1:
                z_crop -= 114.0
                z_crop /= 58.0
        self.model.template(z_crop)

    def track(self, img, hp=None):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        # print("track:",self.center_pos,img.mean(),round(s_x))
        x_crop,im_patchshow = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        if cfg.TRAIN.DATA_NORMALIZE:
            if cfg.TRAIN.DATA_NORMALIZE_MODE==0:
                x_crop /= 255.0
                x_crop -= self.mean
                x_crop /= self.std
            elif cfg.TRAIN.DATA_NORMALIZE_MODE==1:
                x_crop -= 114.0
                x_crop /= 58.0

        outputs = self.model.track(x_crop)
        cls = outputs['cls'].clone().sigmoid()[:,1,:,:].squeeze().data.cpu().numpy()
        cls_show = cls*255
        cls_show = cls_show.astype(np.uint8)
        cv2.namedWindow("cls", cv2.NORM_MINMAX)
        cv2.imshow("cls", cls_show)
        head1 = cv2.resize(cls_show, (cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE))
        heat_img = cv2.applyColorMap(head1, cv2.COLORMAP_JET)
        add_img = cv2.addWeighted(im_patchshow, 0.5, heat_img, 0.5, 0)
        cv2.namedWindow("cls_show", cv2.NORM_MINMAX)
        cv2.imshow("cls_show", add_img)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        # penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        # pscore = pscore * (1 - hp['window_influence']) + \
        #     self.window * hp['window_influence']
        best_idx = np.argmax(pscore)
        pred_bboxbest = pred_bbox[:, best_idx]
        # print("---------------------------------", pred_bboxbest,pred_bboxbest[2]-pred_bboxbest[0],pred_bboxbest[3]-pred_bboxbest[1])
        bbox = pred_bbox[:, best_idx] / scale_z

        s = penalty[best_idx] * score[best_idx]
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        # lr = penalty[best_idx] * score[best_idx] * hp['lr']

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': s
               }
