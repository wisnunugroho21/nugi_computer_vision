import torch.nn as nn
import torch

from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d
from model.component.covnet_attention import CovnetAttention

class ExtractEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ExtractEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_out, dim_out, kernel_size = 3, stride = 1, padding = 1),
        )

        self.conv31 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_out, dim_out, kernel_size = 1),
            nn.GELU(),
            nn.BatchNorm2d(dim_out)
        )

        self.conv32 = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm2d(dim_out),
            DepthwiseSeparableConv2d(dim_out, dim_out, kernel_size = 1),
            nn.BatchNorm2d(dim_out),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = x1 + x

        x2 = self.conv2(x1)
        x2 = x2 + x1

        x31 = self.conv31(x2)
        x32 = self.conv3(x31)
        x32 = x32 + x31
        x33 = self.conv32(x32)

        x3 = x33 + x2

        return x3