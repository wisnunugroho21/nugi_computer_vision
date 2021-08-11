import torch.nn as nn
import torch

from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d

class CovnetAttention(nn.Module):
    def __init__(self, dim, k):
        super(CovnetAttention, self).__init__()

        self.conv_value = nn.Sequential(
            DepthwiseSeparableConv2d(dim, dim, kernel_size = k, stride = k),
            nn.ELU()
        )
        self.conv_key   = nn.Sequential(
            DepthwiseSeparableConv2d(dim, dim, kernel_size = k, stride = k),
            nn.ELU()
        )
        self.conv_query = nn.Sequential(
            DepthwiseSeparableConv2d(dim, dim, kernel_size = 1),
            nn.ELU()
        )
        self.att    = nn.MultiheadAttention(dim, 1, batch_first = True)

    def forward(self, x):
        b, c, h, w = x.shape

        value   = self.conv_value(x).flatten(2).transpose(1, 2)
        key     = self.conv_key(x).flatten(2).transpose(1, 2)
        query   = self.conv_query(x).flatten(2).transpose(1, 2)
        
        x       = self.att(query, key, value)[0]        
        x       = x.transpose(1, 2).reshape(b, c, h, w)

        return x