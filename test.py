import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda:0')
PATH = './deeplabv4_net.pth'

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
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = rate, dilation = rate, bias = False),            
            nn.ReLU(),
            nn.BatchNorm2d(dim_in)
		)

        self.conv1  = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU()
        )
        self.bn1    = nn.BatchNorm2d(dim_out)

        self.conv2  = nn.Sequential(
            DepthwiseSeparableConv2d(dim_out, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.spatial_atrous(x)

        x1 = self.conv1(x)
        x1 = x + x1
        x1 = self.bn1(x1)

        x2 = self.conv2(x1)
        x2 = x1 + x2

        return x2

class SpatialEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SpatialEncoder, self).__init__()

        self.extractor1 = SpatialAtrousExtractor(dim_in, dim_in, 0)
        self.extractor2 = SpatialAtrousExtractor(dim_in, dim_in, 4)
        self.extractor3 = SpatialAtrousExtractor(dim_in, dim_in, 8)
        self.extractor4 = SpatialAtrousExtractor(dim_in, dim_in, 16)

        self.bn = nn.BatchNorm2d(dim_in)

        self.comb_extractor = nn.Sequential(            
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_in)            
        )

        self.change_channel_extractor = nn.Sequential(            
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out)            
        )

        self.downsampler = nn.Sequential(            
            DepthwiseSeparableConv2d(dim_out, dim_out, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out)            
        )

    def forward(self, x):
        x1 = self.extractor1(x)
        x2 = self.extractor1(x)
        x3 = self.extractor1(x)
        x4 = self.extractor1(x)

        xout = x1 + x2 + x3 + x4
        xout = self.bn(xout)
        xout = self.comb_extractor(xout)
        xout = self.change_channel_extractor(xout)
        xout = self.downsampler(xout)

        return xout

class DownsamplerEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DownsamplerEncoder, self).__init__()

        self.downsampler1 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_in),
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out)
        )

        self.downsampler2 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_in),
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out)
        )

    def forward(self, x):
        x1 = self.downsampler1(x)
        x2 = self.downsampler2(x)
        return x1 + x2

class ExtractEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ExtractEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_in),
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_in)            
        )

        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out),
            DepthwiseSeparableConv2d(dim_out, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = x + x1

        x2 = self.conv2(x1)
        x2 = x1 + x2

        return x2

class Deeplabv4(nn.Module):
    def __init__(self, num_classes = 3):
        super(Deeplabv4, self).__init__()

        self.spatialenc = SpatialEncoder(3, 256)

        self.enc1 = DownsamplerEncoder(256, 256)
        self.enc2 = ExtractEncoder(256, 256)
        self.enc3 = DownsamplerEncoder(256, 256)
        self.enc4 = ExtractEncoder(256, 256)
        self.enc5 = DownsamplerEncoder(256, 256)
        self.enc6 = ExtractEncoder(256, 256)

        self.back_channel_extractor = nn.Sequential(
            DepthwiseSeparableConv2d(256, num_classes, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        )

        self.upsample1          = nn.UpsamplingBilinear2d(scale_factor = 4)
        self.upsample_conv_1    = nn.Sequential(
            DepthwiseSeparableConv2d(num_classes, num_classes, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_classes)
        )

        self.upsample2          = nn.UpsamplingBilinear2d(scale_factor = 4)
        self.upsample_conv_2    = nn.Sequential(
            DepthwiseSeparableConv2d(num_classes, num_classes, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_classes)
        )

    def forward(self, x):
        x   = self.spatialenc(x)

        x   = self.enc1(x)
        x   = self.enc2(x)
        x   = self.enc3(x)
        x   = self.enc4(x)
        x   = self.enc5(x)
        x   = self.enc6(x)

        x   = self.back_channel_extractor(x)

        x   = self.upsample1(x)
        x1  = self.upsample_conv_1(x)
        x1  = x + x1

        x1  = self.upsample2(x1)
        x2  = self.upsample_conv_2(x1)
        x2  = x1 + x2

        return x2

class ImageDataset(data.Dataset):
    def __init__(self, transforms1 = None, transforms2 = None):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

        self.imgs   = list(sorted(os.listdir(os.path.join('', 'dataset/images'))))
        self.masks  = list(sorted(os.listdir(os.path.join('', 'dataset/annotations/trimaps'))))

    def __getitem__(self, idx):        
        img_path    = os.path.join('', 'dataset/images', self.imgs[idx])
        mask_path   = os.path.join('', 'dataset/annotations/trimaps', self.masks[idx])
        
        img         = Image.open(img_path).convert("RGB")
        masks       = Image.open(mask_path)

        if self.transforms1 is not None:
            img     = self.transforms1(img)

        if self.transforms2 is not None:    
            masks   = self.transforms2(masks)
        
        img     = torch.FloatTensor(np.array(img))
        img     = img.transpose(1, 2).transpose(0, 1) / 255.0
        
        masks   = torch.LongTensor(np.array(masks))
        masks   = masks.squeeze(0) - masks.min()

        return img, masks

    def __len__(self):
        return len(self.imgs)

def display(disImg):
    disImg = disImg.detach().numpy()
    plt.imshow(disImg)
    plt.show()

transform1 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

transform2 = transforms.Compose([
    transforms.Resize((128, 128))
])

dataset     = ImageDataset(transform1, transform2)
trainloader = data.DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 1)

net         = Deeplabv4(num_classes = 3).to(device)
net.load_state_dict(torch.load(PATH))
net.eval()

inputs, labels = dataset[0]
inputs = inputs.unsqueeze(0).to(device)

outputs = net(inputs)
z = inputs[0].transpose(0, 1).transpose(1, 2)
display(z.cpu())

x = nn.functional.softmax(outputs[0], 0)
x = x.argmax(0)
display(x.cpu())