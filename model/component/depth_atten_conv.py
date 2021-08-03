import torch.nn as nn
import torch

class DepthAttentConv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, stride = 1, padding = 0, dilation = 1, bias = True, depth_multiplier = 1):
        super(DepthAttentConv, self).__init__()

        self.depth_value    = nn.Conv2d(nin, nin * depth_multiplier, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias, groups = nin)
        self.depth_key      = nn.Conv2d(nin, nin * depth_multiplier, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias, groups = nin)
        self.depth_query    = nn.Conv2d(nin, nin * depth_multiplier, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias, groups = nin)

        self.att            = nn.MultiheadAttention(nin, 1)

        self.point          = nn.Conv2d(nin * depth_multiplier, nout, kernel_size = 1, bias = bias)        

    def forward(self, x):
        value   = self.depth_value(x)
        key     = self.depth_key(x)
        query   = self.depth_query(x)

        b, c, h, w = value.shape

        value   = value.flatten(2).transpose(1, 2).transpose(0, 1)
        key     = key.flatten(2).transpose(1, 2).transpose(0, 1)
        query   = query.flatten(2).transpose(1, 2).transpose(0, 1)

        x       = self.att(query, key, value, need_weights = False)[0]
        x       = x.transpose(0, 1).transpose(1, 2).reshape(b, c, h, w)

        return self.point(x)

class PointAttentConv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, stride = 1, padding = 0, dilation = 1, bias = True, depth_multiplier = 1):
        super(PointAttentConv, self).__init__()

        self.point_value    = nn.Conv2d(nin, nout * depth_multiplier, kernel_size = 1, bias = bias)
        self.point_key      = nn.Conv2d(nin, nout * depth_multiplier, kernel_size = 1, bias = bias)
        self.point_query    = nn.Conv2d(nin, nout * depth_multiplier, kernel_size = 1, bias = bias)

        self.att            = nn.MultiheadAttention(nout * depth_multiplier, 1)

        self.depth          = nn.Conv2d(nout * depth_multiplier, nout, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = nout, bias = bias)        

    def forward(self, x):
        value   = self.point_value(x)
        key     = self.point_key(x)
        query   = self.point_query(x)

        b, c, h, w = value.shape

        value   = value.flatten(2).transpose(1, 2).transpose(0, 1)
        key     = key.flatten(2).transpose(1, 2).transpose(0, 1)
        query   = query.flatten(2).transpose(1, 2).transpose(0, 1)

        x       = self.att(query, key, value, need_weights = False)[0]
        x       = x.transpose(0, 1).transpose(1, 2).reshape(b, c, h, w)

        return self.depth(x)