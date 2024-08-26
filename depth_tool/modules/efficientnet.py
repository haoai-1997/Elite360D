import math
import torch
from torch import nn
from torch.nn import functional as F
from depth_tool.modules.efficientnet_pytorch import EfficientNet as effNet

class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


class EfficientNet(nn.Module):
    def __init__(self, num_classes, version = 'b5'):
        super(EfficientNet, self).__init__()

        # load pretrained EfficientNet B3
        self.model_ft = effNet.from_pretrained(f'efficientnet-{version}')

        for child in self.model_ft.children():

          for param in child.parameters():
            param.requires_grad = False

        # re-init last conv layer and last fc layer to fit with dataset
        in_channels = self.model_ft._conv_head.in_channels
        out_channels = self.model_ft._conv_head.out_channels
        num_ftrs = self.model_ft._fc.in_features

        self.model_ft._conv_head = Conv2dSamePadding(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), bias=False)
        self.model_ft._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=0.010000000000000009, eps = 0.001)
        self.model_ft._fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model_ft(x)

class EfficientNetB5(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB5, self).__init__()

        # load pretrained EfficientNet B3
        if pretrained == True:
            self.model_ft = effNet.from_pretrained('efficientnet-b5',weights_path="checkpoints/adv-efficientnet-b5-86493f6b.pth", advprop=True)
        else:
            self.model_ft = effNet.from_name('efficientnet-b5')
        del self.model_ft._conv_head
        del self.model_ft._bn1
        del self.model_ft._fc
    def forward(self, x):
        endpoints = self.model_ft.extract_endpoints(x)
        return endpoints