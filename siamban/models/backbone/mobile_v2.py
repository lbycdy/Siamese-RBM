from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from functools import reduce


flagorg = False

if flagorg:

    def conv_bn(inp, oup, stride, padding=1):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, padding, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )


    def conv_1x1_bn(inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )


    class InvertedResidual(nn.Module):
        def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
            super(InvertedResidual, self).__init__()
            self.stride = stride

            self.use_res_connect = self.stride == 1 and inp == oup

            padding = 2 - stride
            if dilation > 1:
                padding = dilation

            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3,
                          stride, padding, dilation=dilation,
                          groups=inp * expand_ratio, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)


    class MobileNetV2(nn.Sequential):
        def __init__(self, width_mult=1.0, used_layers=[3, 5, 7]):
            super(MobileNetV2, self).__init__()

            self.interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1, 1],
                [6, 24, 2, 2, 1],
                [6, 32, 3, 2, 1],
                [6, 64, 4, 2, 1],
                [6, 96, 3, 1, 1],
                [6, 160, 3, 2, 1],
                [6, 320, 1, 1, 1],
            ]
            # 0,2,3,4,6

            self.interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1, 1],
                [6, 24, 2, 2, 1],
                [6, 32, 3, 2, 1],
                [6, 64, 4, 1, 2],
                [6, 96, 3, 1, 2],
                [6, 160, 3, 1, 4],
                [6, 320, 1, 1, 4],
            ]

            self.channels = [24, 32, 96, 320]
            self.channels = [int(c * width_mult) for c in self.channels]

            input_channel = int(32 * width_mult)
            self.last_channel = int(1280 * width_mult) \
                if width_mult > 1.0 else 1280

            self.add_module('layer0', conv_bn(3, input_channel, 2, 0))

            last_dilation = 1

            self.used_layers = used_layers

            for idx, (t, c, n, s, d) in \
                    enumerate(self.interverted_residual_setting, start=1):
                output_channel = int(c * width_mult)

                layers = []

                for i in range(n):
                    if i == 0:
                        if d == last_dilation:
                            dd = d
                        else:
                            dd = max(d // 2, 1)
                        layers.append(InvertedResidual(input_channel,
                                                       output_channel, s, t, dd))
                    else:
                        layers.append(InvertedResidual(input_channel,
                                                       output_channel, 1, t, d))
                    input_channel = output_channel

                last_dilation = d

                self.add_module('layer%d' % (idx), nn.Sequential(*layers))

        def forward(self, x):
            # exit()
            outputs = []
            for idx in range(8):
                name = "layer%d" % idx
                x = getattr(self, name)(x)
                outputs.append(x)
            p0, p1, p2, p3, p4 = [outputs[i] for i in [1, 2, 3, 5, 7]]
            out = [outputs[i] for i in self.used_layers]
            if len(out) == 1:
                return out[0]
            return out



else:
    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


    def conv_bn(inp, oup, stride, padd=1, dilation=1, flagrelu=True, ngroup=1, ksize=3):
        if flagrelu:
            return nn.Sequential(
                nn.Conv2d(inp, oup, ksize, stride, padd, bias=False, groups=ngroup),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inp, oup, ksize, stride, padd, bias=False, groups=ngroup),
                nn.BatchNorm2d(oup),
            )


    def conv_1x1_bn(inp, oup, flagrelu=True, stride=1):
        if flagrelu:
            return nn.Sequential(
                nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    class InvertedResidual(nn.Module):
        def __init__(self, inp, oup, stride, expansion, dilation=1):
            super(InvertedResidual, self).__init__()
            self.stride = stride
            assert stride in [1, 2]
            pad = 2 - stride
            if dilation > 1:
                pad = dilation

            hidden_dim = round(inp * expansion)
            self.use_res_connect = self.stride == 1 and inp == oup

            if expansion == 1:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, groups=hidden_dim, dilation=dilation, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, groups=hidden_dim, dilation=dilation, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )

        def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)


    class MobileNetV2(nn.Module):
        def __init__(self, width_mult=1.0, used_layers=[3, 5, 7],flaglastconv = False):
            super(MobileNetV2, self).__init__()
            self.used_layers = used_layers
            self.in_channels = 3
            alpha = width_mult
            expansion = 6
            input_channel = 32
            last_channel = 1280

            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1, 1],
                [expansion, 24, 2, 2, 1],
                [expansion, 32, 3, 2, 1],
                [expansion, 64, 4, 1, 1],
                [expansion, 96, 3, 1, 1],
                [expansion, 160, 3, 1, 4],
                [expansion, 320, 1, 1, 4],
            ]
            self.blockid2featid_dict = {}
            # building first layer
            input_channel = _make_divisible(input_channel * alpha, 8)
            self.last_channel = _make_divisible(last_channel * alpha, 8) if alpha > 1.0 else last_channel

            self.features = [conv_bn(self.in_channels, input_channel, 2, padd=0)]
            self.blockid2featid_dict[0] = 0
            blockid = 0
            featid = 0
            # building inverted residual blocks
            for t, c, n, s, d in interverted_residual_setting:
                output_channel = _make_divisible(int(c * alpha), 8)
                blockid += 1
                for i in range(n):
                    if i == 0:
                        self.features.append(
                            InvertedResidual(input_channel, output_channel, s, expansion=t, dilation=d))
                    else:
                        self.features.append(
                            InvertedResidual(input_channel, output_channel, 1, expansion=t, dilation=d))
                    input_channel = output_channel

                    featid += 1

                self.blockid2featid_dict[blockid] = featid
            if flaglastconv:
                self.blockid2featid_dict[blockid+1] = featid+1
                self.features.append(conv_1x1_bn(input_channel, self.last_channel))
            self.features = nn.Sequential(*self.features)
            # Initialize weights
            # self._init_weights()

        def forward(self, x, feature_names=None):
            out = []
            feaidstart = 0
            for blockid in self.used_layers:
                featidend = self.blockid2featid_dict[blockid] + 1
                x = reduce(lambda x, n: self.features[n](x), list(range(feaidstart, featidend)), x)
                out.append(x)
                feaidstart = featidend
            if len(out) == 1:
                out = out[0]
            # Output
            return out
        def loadfrompretrain(self,pretrainfile):
            pretrained = torch.load(pretrainfile, map_location=torch.device('cpu'))
            state_dict = self.state_dict()
            key_src = list(pretrained.keys())
            key_dst = list(state_dict.keys())
            for idx, keyd in enumerate(key_dst):
                key = key_src[idx]
                state_dict[keyd] = pretrained[key]
            self.load_state_dict(state_dict)



def mobilenetv2(**kwargs):
    model = MobileNetV2(**kwargs)
    return model

if __name__ == '__main__':
    from torchvision import transforms

    # transform = transforms.Compose([  # [1]
    #     transforms.Resize(256),  # [2]
    #     transforms.CenterCrop(224),  # [3]
    #     transforms.ToTensor(),  # [4]
    #     transforms.Normalize(  # [5]
    #     mean = [0.485, 0.456, 0.406],  # [6]
    # std = [0.229, 0.224, 0.225]  # [7]
    # )])
    net = MobileNetV2(used_layers=[0,1,2,3,4,5,6,7,8],flaglastconv=True)
    for key in net.blockid2featid_dict:
        print(key,net.blockid2featid_dict[key])
    # pretrained = torch.load("/home/ethan/work/pysot/pretrained_models/mobilenet_v2-b0353104.pth")
    # state_dict = net.state_dict()
    # key_src = list(pretrained.keys())
    # key_dst = list(state_dict.keys())
    # for idx, keyd in enumerate(key_dst):
    #     key = key_src[idx]
    #     state_dict[keyd] = pretrained[key]
    #     print(key,pretrained[key].size(),state_dict[keyd].size())

    # keysrc = list(pretrained.keys())
    # state_dict = net.state_dict()
    # for key in state_dict:
    #     print(key)
    # exit()
    # keydst = list(state_dict.keys())
    # keysrc.sort()
    # keysrc.remove("classifier.1.bias")
    # keysrc.remove("classifier.1.weight")
    #
    # keydst.sort()
    # for key in keydst:
    #     if "layer0" in key:
    #         print(key,state_dict[key].size())
    # exit()
    # for i in range(min(len(keysrc),len(keydst))):
    #     kd = keydst[i]
    #     ks = keysrc[i]
    #     sized = state_dict[kd].size()
    #     sizes = pretrained[ks].size()
    #     # if sized!=sizes:
    #     print(ks,kd,sizes,sized)

    # print(len(keysrc),len(keydst))
    # for key in net.state_dict():
    #     print(key)
    # exit()
    # from torch.autograd import Variable
    # tensor = Variable(torch.Tensor(1, 3, 255, 255)).cuda()
    #
    # net = net.cuda()
    #
    # out = net(tensor)
    #
    # for i, p in enumerate(out):
    #     print(i, p.size())
