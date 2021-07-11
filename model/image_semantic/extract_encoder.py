import torch.nn as nn
from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d

class ExtractEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ExtractEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ELU(),
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ELU(),                    
        )

        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ELU(),
            DepthwiseSeparableConv2d(dim_out, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ELU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = x + x1

        x2 = self.conv2(x1)
        x2 = x1 + x2

        return x2