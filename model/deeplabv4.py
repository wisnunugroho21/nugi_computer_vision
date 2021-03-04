import torch
import torch.nn as nn

device = torch.device('cuda:0')

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, stride = 1, padding = 0, dilation = 1, bias = False, depth_multiplier = 1):
        super(DepthwiseSeparableConv2d, self).__init__()

        self.nn_layer = nn.Sequential(
            nn.Conv2d(nin, nin * depth_multiplier, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias, groups = nin),
            nn.Conv2d(nin * depth_multiplier, nout, kernel_size = 1, bias = bias)
        )
    
    def forward(self, x):        
        return self.nn_layer(x)

class SpatialAtrousExtractor(nn.Module):
    def __init__(self, dim_in, dim_out, rate):
        super(SpatialAtrousExtractor, self).__init__()        

        self.spatial_atrous = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = rate, dilation = rate, bias = False),            
            nn.ReLU()
		)

    def forward(self, x):
        x = self.spatial_atrous(x)
        return x

class SpatialEncoder(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, ):
        super(SpatialEncoder, self).__init__()        

        self.extractor1 = SpatialAtrousExtractor(dim_in, dim_mid, 0)
        self.extractor2 = SpatialAtrousExtractor(dim_in, dim_mid, 2)
        self.extractor3 = SpatialAtrousExtractor(dim_in, dim_mid, 8)
        self.extractor4 = SpatialAtrousExtractor(dim_in, dim_mid, 16)

        self.out = nn.Sequential(
            DepthwiseSeparableConv2d(4 * dim_mid, dim_out, kernel_size = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out),
        )

    def forward(self, x):
        x1 = self.extractor1(x)
        x2 = self.extractor1(x)
        x3 = self.extractor1(x)
        x4 = self.extractor1(x)      

        xout = torch.cat((x1, x2, x3, x4), 1)
        xout = self.out(xout)

        return xout

class DownsamplerEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DownsamplerEncoder, self).__init__()

        self.downsampler1 = nn.Sequential(            
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.ReLU(),
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.ReLU()
        )
        self.downsampler2 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 8, stride = 4, padding = 2, bias = False),
            nn.ReLU()
        )

        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):        
        x1      = self.downsampler1(x)
        x2      = self.downsampler2(x)

        xout    = self.bn(x1 + x2)
        return xout

class ExtractEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ExtractEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),                    
        )

        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            DepthwiseSeparableConv2d(dim_out, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU()
        )

        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = x + x1

        x2 = self.conv2(x1)
        x2 = self.bn(x1 + x2)

        return x2

class Deeplabv4(nn.Module):
    def __init__(self, num_classes = 3):
        super(Deeplabv4, self).__init__()

        self.spatialenc = SpatialEncoder(3, 16, 256)

        self.enc1 = ExtractEncoder(256, 256)
        self.enc2 = ExtractEncoder(256, 256)
        self.enc3 = DownsamplerEncoder(256, 256)
        self.enc4 = ExtractEncoder(256, 256)
        self.enc5 = ExtractEncoder(256, 256)
        self.enc6 = DownsamplerEncoder(256, 256)
        self.enc7 = ExtractEncoder(256, 256)
        self.enc8 = ExtractEncoder(256, 256)       

        self.back_channel_extractor = nn.Sequential(
            DepthwiseSeparableConv2d(256, 16, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            DepthwiseSeparableConv2d(16, num_classes, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(num_classes),
        )

        self.upsample1          = nn.UpsamplingBilinear2d(scale_factor = 4)
        self.upsample_conv_1    = nn.Sequential(
            DepthwiseSeparableConv2d(num_classes, num_classes, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.ReLU()
        )

        self.upsample2          = nn.UpsamplingBilinear2d(scale_factor = 4)
        self.upsample_conv_2    = nn.Sequential(
            DepthwiseSeparableConv2d(num_classes, num_classes, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.ReLU()
        )

    def forward(self, x):
        x   = self.spatialenc(x)

        x   = self.enc1(x)
        x   = self.enc2(x)
        x   = self.enc3(x)
        x   = self.enc4(x)
        x   = self.enc5(x)
        x   = self.enc6(x)
        x   = self.enc7(x)
        x   = self.enc8(x)

        x   = self.back_channel_extractor(x)

        x   = self.upsample1(x)
        x1  = self.upsample_conv_1(x)
        x1  = x + x1
        
        x1  = self.upsample2(x1)
        x2  = self.upsample_conv_2(x1)
        x2  = x1 + x2

        return x2