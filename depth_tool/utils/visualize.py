import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plot
import cv2
from PIL import Image
#from torchsummary import summary
from thop import profile
__all__ = ['show_flops_params']

@torch.no_grad()
def show_flops_params(model, device, input_shape=[1, 3, 512, 1024],input_shape1=[1, 256*20, 3],input_shape2=[1, 256*20, 3]):
    input = torch.randn(*input_shape).to(torch.device(device))
    input1 = torch.randn(*input_shape1).to(torch.device(device))
    input2 = torch.randn(*input_shape2).to(torch.device(device))

    flops, params = profile(model, inputs=(input,input1,input2), verbose=False)

    print('{} flops: {:.3f}G input shape is {}, params: {:.3f}M'.format(
        model.__class__.__name__, flops / 1000000000, input_shape[1:], params / 1000000))
