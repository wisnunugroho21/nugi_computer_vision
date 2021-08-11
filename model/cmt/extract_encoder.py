import torch.nn as nn
import torch

from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d
from model.component.covnet_attention import CovnetAttention

class ExtractEncoder(nn.Module):
    def __init__(self, dim, k):
        super(ExtractEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1),
            nn.ELU()
        )
        
        self.att = CovnetAttention(dim, k)        

        self.conv21 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 1, bias = False),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

        self.conv2  = DepthwiseSeparableConv2d(dim, dim, kernel_size = 3, stride = 1, padding = 1)

        self.conv22 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size = 1, bias = False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x1 = self.conv1(x)
        x1 = x1 + x

        x2 = torch.nn.functional.layer_norm(x1, (c, h, w))
        x2 = self.att(x2)
        x2 = x2 + x1

        x3  = torch.nn.functional.layer_norm(x2, (c, h, w))
        x31 = self.conv21(x3)
        x32 = self.conv2(x31)
        x32 = x32 + x31
        x33 = self.conv22(x32)

        x3 = x33 + x2

        return x3