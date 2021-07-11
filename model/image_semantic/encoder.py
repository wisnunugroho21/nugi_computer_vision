import torch.nn as nn
from model.component.atrous_spatial_pyramid_conv2d import AtrousSpatialPyramidConv2d
from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d
from model.image_semantic.extract_encoder import ExtractEncoder
from model.image_semantic.downsampler_encoder import DownsamplerEncoder

class Encoder(nn.Module):
    def __init__(self, num_classes = 3):
        super(Encoder, self).__init__()

        self.spatialenc = AtrousSpatialPyramidConv2d(3, 16, 64)

        # self.enc0 = nn.Sequential(
        #     DepthwiseSeparableConv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
        #     nn.ELU()
        # )

        self.enc1 = ExtractEncoder(64, 64)
        self.enc2 = ExtractEncoder(64, 64)
        self.enc3 = DownsamplerEncoder(64, 128)
        self.enc4 = ExtractEncoder(128, 128)
        self.enc5 = ExtractEncoder(128, 128)
        self.enc6 = DownsamplerEncoder(128, 256)
        self.enc7 = ExtractEncoder(256, 256)
        self.enc8 = ExtractEncoder(256, 256)

    def forward(self, x):
        x   = self.spatialenc(x)
        # x   = self.enc0(x)

        x   = self.enc1(x)
        x   = self.enc2(x)
        x   = self.enc3(x)
        x   = self.enc4(x)
        x   = self.enc5(x)
        x   = self.enc6(x)
        x   = self.enc7(x)
        x   = self.enc8(x)

        return x