import copy
import math

import torch
from torch import nn
from torch.nn import Sequential

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import IntBias

from brevitas_examples.imagenet_classification.models.common import CommonIntActQuant, CommonUintActQuant
from brevitas_examples.imagenet_classification.models.common import CommonIntWeightPerChannelQuant, CommonIntWeightPerTensorQuant

from ultralytics.nn.modules import DFL
from ultralytics.nn.modules.head import Detect


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
    

# -----------------QUANTIZED -----------

class QuantConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            weight_bit_width=8,
            act_bit_width=8,
            act=True,
            padding=None,
            groups=1,
            dilation=1):
        super(QuantConv, self).__init__()
        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=autopad(kernel_size, padding, dilation),
            groups=groups,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-5)
        self.act = act if isinstance(act, nn.Module) else QuantReLU(
                                                                    act_quant=CommonUintActQuant,
                                                                    bit_width=act_bit_width,
                                                                    per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                                                                    scaling_per_channel=True,
                                                                    return_quant_tensor=False)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# --------- yolov8zone

class QuantC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, weight_bit_width=8, act_bit_width=8, act=True, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.common_act = QuantReLU(
                                    act_quant=CommonUintActQuant,
                                    bit_width=act_bit_width,
                                    scaling_per_channel=False,
                                    return_quant_tensor=False)
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = QuantConv(c1, 2 * self.c, 1, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, act=self.common_act)
        self.cv2 = QuantConv((2 + n) * self.c, c2, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, act=act)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(QuantBottleneck(self.c, self.c, shortcut=shortcut, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, cv2_act=self.common_act, g=g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class QuantBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, weight_bit_width=8, act_bit_width=8, cv2_act=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = QuantConv(c1, c_, k[0], 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.cv2 = QuantConv(c_, c2, k[1], 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, act=cv2_act, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class QuantSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5, weight_bit_width=8, act_bit_width=8, act=True):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = QuantConv(c1, c_, 1, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.cv2 = QuantConv(c_ * 4, c2, 1, 1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
    

class QuantDetect(Detect):
    """YOLOv8 Detect head for detection models."""

    finn_export = False

    def __init__(self, nc=80, weight_bit_width=8, act_bit_width=8, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__(nc, ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(QuantConv(x, c2, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                          QuantConv(c2, c2, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                          QuantConv2d(c2, 4 * self.reg_max, 1, weight_quant=CommonIntWeightPerChannelQuant, weight_bit_width=weight_bit_width)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(QuantConv(x, c3, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                          QuantConv(c3, c3, 3, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                          QuantConv2d(c3, self.nc, 1, weight_quant=CommonIntWeightPerChannelQuant, weight_bit_width=weight_bit_width)) for x in ch
        )

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)
        # print('detect input:', [xi.shape for xi in x])

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training or self.finn_export:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)


class DummyDetect(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x