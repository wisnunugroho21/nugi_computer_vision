import torch.nn as nn
from model.component.atrous_spatial_pyramid_conv2d import AtrousSpatialPyramidConv2d
from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d
from model.covnet_attention.extract_encoder import ExtractEncoder
from model.image_semantic_segmentation.downsampler_encoder import DownsamplerEncoder

from model.component.covnet_attention import CovnetAttention

class Encoder(nn.Module):
    def __init__(self, num_classes = 3):
        super(Encoder, self).__init__()

        """ self.spatialenc = AtrousSpatialPyramidConv2d(3, 16, 64)

        self.enc1 = ExtractEncoder(64, 64)
        self.enc2 = ExtractEncoder(64, 64)
        self.enc3 = DownsamplerEncoder(64, 128)
        self.enc4 = ExtractEncoder(128, 128)
        self.enc5 = ExtractEncoder(128, 128)
        self.enc6 = DownsamplerEncoder(128, 256)
        self.enc7 = ExtractEncoder(256, 256)
        self.enc8 = ExtractEncoder(256, 256) """

        """ self.net = nn.Sequential(
            DepthwiseSeparableConv2d(3, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.GELU(),
            nn.BatchNorm2d(16),
            ExtractEncoder(16, 16),
            DepthwiseSeparableConv2d(16, 32, kernel_size = 2, stride = 2),
            nn.ReLU(),
            ExtractEncoder(32, 32),
            DepthwiseSeparableConv2d(32, 64, kernel_size = 2, stride = 2),
            nn.ReLU(),
            ExtractEncoder(64, 64),
            DepthwiseSeparableConv2d(64, 128, kernel_size = 2, stride = 2),
            nn.ReLU(),
            ExtractEncoder(128, 128),
            DepthwiseSeparableConv2d(128, 256, kernel_size = 2, stride = 2),
            nn.ReLU(),
        ) """

        self.net = nn.Sequential(            
            DepthwiseSeparableConv2d(3, 16, kernel_size = 1),
            nn.GELU(),
            nn.BatchNorm2d(16),
            ExtractEncoder(16, 16),
            DownsamplerEncoder(16, 32),
            ExtractEncoder(32, 32),
            DownsamplerEncoder(32, 64),
            ExtractEncoder(64, 64),
            DownsamplerEncoder(64, 128),
            ExtractEncoder(128, 128),
            DownsamplerEncoder(128, 256),
            ExtractEncoder(256, 256),
        )

    def forward(self, x):
        """ x   = self.spatialenc(x)
        # x   = self.enc0(x)

        x   = self.enc1(x)
        x   = self.enc2(x)
        x   = self.enc3(x)
        x   = self.enc4(x)
        x   = self.enc5(x)
        x   = self.enc6(x)
        x   = self.enc7(x)
        x   = self.enc8(x) """

        return self.net(x)