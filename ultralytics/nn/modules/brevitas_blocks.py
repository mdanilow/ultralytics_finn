import math

import torch
from torch import nn
from torch.nn import Sequential

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import IntBias

from brevitas_examples.imagenet_classification.models.common import CommonIntActQuant, CommonUintActQuant
from brevitas_examples.imagenet_classification.models.common import CommonIntWeightPerChannelQuant, CommonIntWeightPerTensorQuant

from ultralytics.nn.modules import DFL
from ultralytics.utils.tal import make_anchors, dist2bbox


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
    

class QuantDetect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    finn_export = False
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, weight_bit_width=8, act_bit_width=8, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
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
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

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

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        # print('inference', [xi.shape for xi in x])
        shape = x[0].shape  # BCHW
        # print('x precat:', [xi.view(shape[0], self.no, -1).shape for xi in x])
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        # print('x cat:', x_cat.shape)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
            # print('anchors, strides:', self.anchors.shape, self.strides.shape)
            # # torch.set_printoptions(profile="full")
            # print(self.anchors)
            # # torch.set_printoptions(profile="default")
            # print(self.strides)

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes the predictions obtained from a YOLOv10 model.

        Args:
            preds (torch.Tensor): The predictions obtained from the model. It should have a shape of (batch_size, num_boxes, 4 + num_classes).
            max_det (int): The maximum number of detections to keep.
            nc (int, optional): The number of classes. Defaults to 80.

        Returns:
            (torch.Tensor): The post-processed predictions with shape (batch_size, max_det, 6),
                including bounding boxes, scores and cls.
        """
        assert 4 + nc == preds.shape[-1]
        boxes, scores = preds.split([4, nc], dim=-1)
        max_scores = scores.amax(dim=-1)
        max_scores, index = torch.topk(max_scores, min(max_det, max_scores.shape[1]), axis=-1)
        index = index.unsqueeze(-1)
        boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
        scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

        # NOTE: simplify but result slightly lower mAP
        # scores, labels = scores.max(dim=-1)
        # return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

        scores, index = torch.topk(scores.flatten(1), max_det, axis=-1)
        labels = index % nc
        index = index // nc
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))

        return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)


class DummyDetect(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x