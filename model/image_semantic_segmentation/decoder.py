import torch.nn as nn
from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d

class Decoder(nn.Module):
    def __init__(self, num_classes = 3):
        super(Decoder, self).__init__()       

        self.back_channel_extractor = nn.Sequential(
            DepthwiseSeparableConv2d(256, num_classes, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
        )

        self.upsample1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.UpsamplingBilinear2d(scale_factor = 2),
        )

        self.upsample_fixer1 = nn.Sequential(
            DepthwiseSeparableConv2d(num_classes, num_classes, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
        )

        self.upsample2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.UpsamplingBilinear2d(scale_factor = 2),
        )        

        self.upsample_fixer2 = nn.Sequential(
            DepthwiseSeparableConv2d(num_classes, num_classes, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
        )        

    def forward(self, x):
        x   = self.back_channel_extractor(x)

        x1  = self.upsample1(x)
        x11 = self.upsample_fixer1(x1)
        x1  = x11 + x1

        x2  = self.upsample2(x1)
        x21 = self.upsample_fixer2(x2)
        x2  = x21 + x2

        return x2