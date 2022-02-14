# This file contains modules common to various models

import math
import numpy as np
import torch
import torch.nn as nn

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model

    def forward(self, x, size=640, augment=False, profile=False):
        # supports inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   opencv:     x = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:        x = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:      x = np.zeros((720,1280,3))  # HWC
        #   torch:      x = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:   x = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(x, torch.Tensor):  # torch
            return self.model(x.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        if not isinstance(x, list):
            x = [x]
        shape0, shape1 = [], []  # image and inference shapes
        batch = range(len(x))  # batch size
        for i in batch:
            x[i] = np.array(x[i])[:, :, :3]  # up to 3 channels if png
            s = x[i].shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(x[i], new_shape=shape1, auto=False)[0] for i in batch]  # pad
        x = np.stack(x, 0) if batch[-1] else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        x = self.model(x, augment, profile)  # forward
        x = non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in batch:
            if x[i] is not None:
                x[i][:, :4] = scale_coords(shape1, x[i][:, :4], shape0[i])
        return x


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


# Experiements for project

class CSPBottleneck3Conv(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CSPBottleneck3Conv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


# initialize transformer block with a CSP block
class TF(CSPBottleneck3Conv):
    # initialize transformer block
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        # added a CSP layer
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # create transforemer block
        self.m = TFBlock(c_, c_, 4, n)


# transformer layer class
class TFLayer(nn.Module):
    # initialize layer
    def __init__(self, c, num_heads):
        super().__init__()
        # transformer parameters and components
        self.q = nn.Linear(c, c, bias=False)  # query target
        self.k = nn.Linear(c, c, bias=False)  # keys source
        self.v = nn.Linear(c, c, bias=False)  # values source

        # from the paper
        # multihead structure
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        # normalization structure
        self.ln1 = nn.LayerNorm(c)
        self.ln2 = nn.LayerNorm(c)
        # linear structures
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, input):
        # normalisation of input
        ln1_output = self.ln1(input)
        # using the keys, query, and values as input to the
        # multihead attention machanism
        ma_output = self.ma(self.q(ln1_output), self.k(ln1_output), self.v(ln1_output))[0] + input
        # normalisation of multihead output
        ln2_output = self.ln2(ma_output)
        # linearization of normalisation output
        output = self.fc2(self.fc1(ln2_output)) + ln2_output
        # return output
        return output


# transformer block class
# create a block of transformer based on the layers
class TFBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()

        # standard convolution if input
        # and output are not equal
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        # linearisation of input
        self.linear = nn.Linear(c2, c2)
        # create sequence of multiple layer of transformer
        self.tr = nn.Sequential(*[TFLayer(c2, num_heads) for _ in range(num_layers)])
        # set output size
        self.c2 = c2

    def forward(self, input):
        if self.conv is not None:
            input = self.conv(input)
        # rshape of tensor
        b, _, w, h = input.shape
        p = input.flatten(2)
        # reformat tensor
        p = p.unsqueeze(0)
        # transpose the tensor
        p = p.transpose(0, 3)
        # remove dimensition of size 3
        p = p.squeeze(3)
        # linear
        e = self.linear(p)
        # add linear value
        input = p + e

        # sequential
        input = self.tr(input)
        # reformat tensor
        input = input.unsqueeze(3)
        # transpose
        input = input.transpose(0, 3)
        # reshape
        input = input.reshape(b, self.c2, w, h)
        # return output
        return input