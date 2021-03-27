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
            nn.Conv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = rate, dilation = rate)
		)

    def forward(self, x):
        x = self.spatial_atrous(x)
        return x

class SpatialEncoder(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, ):
        super(SpatialEncoder, self).__init__()        

        self.extractor1 = SpatialAtrousExtractor(dim_in, dim_mid, 0)
        self.extractor2 = SpatialAtrousExtractor(dim_in, dim_mid, 2)
        self.extractor3 = SpatialAtrousExtractor(dim_in, dim_mid, 6)

        self.out = nn.Sequential(
            nn.Conv2d(3 * dim_mid, dim_out, kernel_size = 1)
        )

    def forward(self, x):
        x1 = self.extractor1(x)
        x2 = self.extractor2(x)
        x3 = self.extractor3(x) 

        xout = torch.cat((x1, x2, x3), 1)
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
        x1 = self.bn(x + x1)

        x2 = self.conv2(x1)
        x2 = self.bn(x1 + x2)

        return x2

class Encoder(nn.Module):
    def __init__(self, num_classes = 3):
        super(Encoder, self).__init__()

        self.spatialenc = SpatialEncoder(3, 3, 64)

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

        x   = self.enc1(x)
        x   = self.enc2(x)
        x   = self.enc3(x)
        x   = self.enc4(x)
        x   = self.enc5(x)
        x   = self.enc6(x)
        x   = self.enc7(x)
        x   = self.enc8(x)

        return x

class Decoder(nn.Module):
    def __init__(self, num_classes = 3):
        super(Decoder, self).__init__()       

        self.back_channel_extractor = nn.Sequential(
            DepthwiseSeparableConv2d(256, num_classes, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
        )

        self.upsample1  = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU()
        )

        self.upsample2  = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU()
        )

    def forward(self, x):
        x   = self.back_channel_extractor(x)

        x   = self.upsample1(x)        
        x   = self.upsample2(x)

        return x