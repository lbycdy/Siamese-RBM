# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import numpy as np

from siamban.utils.bbox import corner2center, center2corner


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
        return True

# score_size = 25
anchors = Anchors(8,[1,],[11])
anchor = anchors.anchors
anchors.generate_all_anchors(80,11)
# x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
# anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
# print(anchor.shape)
# total_stride = anchors.stride
# anchor_num = anchor.shape[0]
# print(np.tile(anchor, score_size * score_size).shape)
# anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
# print(anchor.shape)
# print(anchor[:5])
# print(anchor[5:10])
# ori = - (score_size // 2) * total_stride
# xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
#                  [ori + total_stride * dy for dy in range(score_size)])
# xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
# np.tile(yy.flatten(), (anchor_num, 1)).flatten()
# anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
# print(anchor[:10]+128)
