import torch
import torch.nn as nn
from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d

class Downsampler(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Downsampler, self).__init__()

        self.downsampler1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size = 4, stride = 2, padding = 1, groups = dim_in)
        )
        self.downsampler2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size = 2, stride = 2, groups = dim_in),
        )

        self.out = nn.Sequential(
            nn.Conv2d(2 * dim_out, dim_out, kernel_size = 1),
            nn.ELU()
        ) 

    def forward(self, x):        
        x1      = self.downsampler1(x)
        x2      = self.downsampler2(x)

        xout    = torch.cat((x1, x2), 1)
        xout    = self.out(xout)

        return xout