import torch
import torch.nn as nn
from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d

class DownsamplerEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DownsamplerEncoder, self).__init__()

        self.downsampler1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = 4, stride = 2, padding = 1, groups = dim_in, bias = False),
            nn.Conv2d(dim_in, dim_in, kernel_size = 4, stride = 2, padding = 1, groups = dim_in, bias = False)
        )
        self.downsampler2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = 8, stride = 4, padding = 2, groups = dim_in, bias = False),
        )

        self.out = nn.Sequential(
            nn.Conv2d(2 * dim_in, dim_out, kernel_size = 1),
            nn.ELU()
        )

    def forward(self, x):        
        x1      = self.downsampler1(x)
        x2      = self.downsampler2(x)

        xout    = torch.cat((x1, x2), 1)
        xout    = self.out(xout)

        return xout