import torch.nn as nn
import torch

from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d

class CovnetAttention(nn.Module):
    def __init__(self, dim):
        super(CovnetAttention, self).__init__()

        self.att    = nn.MultiheadAttention(dim, 1)

        self.conv_value = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 1),
            nn.ReLU(),
        )     

        self.conv_key = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 1),
            nn.ReLU(),
        )

        self.conv_query = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 1),
            nn.ReLU(),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        value   = self.conv_value(x).flatten(2).transpose(1, 2).transpose(0, 1)
        key     = self.conv_key(x).flatten(2).transpose(1, 2).transpose(0, 1)
        query   = self.conv_query(x).flatten(2).transpose(1, 2).transpose(0, 1)

        x       = self.att(query, key, value, need_weights = False)[0]
        x       = x.transpose(0, 1).transpose(1, 2).reshape(b, c, h, w)

        return x