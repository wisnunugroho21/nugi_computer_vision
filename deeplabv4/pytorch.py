import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from PIL import Image

import torch.optim as optim

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, stride = 1, padding = 0, dilation = 1, bias = False, depth_multiplier = 1, activate_first = True):
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
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 3, stride = 1, bias = False),
            nn.ReLU()
        )
        self.bn1    = nn.BatchNorm2d(dim_out)

        self.conv2  = nn.Sequential(
            DepthwiseSeparableConv2d(dim_out, dim_out, kernel_size = 3, stride = 1, bias = False),
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

class FirstSpatialEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FirstSpatialEncoder, self).__init__()

        self.extractor1 = SpatialAtrousExtractor(dim_in, dim_out, 0)
        self.extractor2 = SpatialAtrousExtractor(dim_in, dim_out, 2)
        self.extractor3 = SpatialAtrousExtractor(dim_in, dim_out, 4)
        self.extractor4 = SpatialAtrousExtractor(dim_in, dim_out, 6)

        self.bn = nn.BatchNorm2d(dim_out)

        self.comb_extractor = nn.Sequential(            
            DepthwiseSeparableConv2d(dim_out, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
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
        xout = self.downsampler(xout)

        return xout

class SecondSpatialEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SecondSpatialEncoder, self).__init__()

        self.extractor1 = SpatialAtrousExtractor(dim_in, dim_out, 0)
        self.extractor2 = SpatialAtrousExtractor(dim_in, dim_out, 2)

        self.bn = nn.BatchNorm2d(dim_out)

        self.comb_extractor = nn.Sequential(            
            DepthwiseSeparableConv2d(dim_out, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
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

        xout = x1 + x2
        xout = self.bn(xout)
        xout = self.comb_extractor(xout)
        xout = self.downsampler(xout)

class DownsamplerEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DownsamplerEncoder, self).__init__()

        self.conv = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_in),
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_in),
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_in)            
        )

        self.downsampler = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out)
        )

    def forward(self, x):
        x1 = self.conv(x)
        x1 = x + x1

        x2 = self.downsampler(x1)
        return x2

class SameDimensionEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SameDimensionEncoder, self).__init__()

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
    def __init__(self):
        super(Deeplabv4, self).__init__()

        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv2d(3, 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv2d(64, 256, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.enc1 = FirstSpatialEncoder(256, 256)
        self.enc2 = SecondSpatialEncoder(256, 256)
        self.enc3 = SameDimensionEncoder(256, 256)
        self.enc4 = DownsamplerEncoder(256, 256)
        self.enc5 = SameDimensionEncoder(256, 256)
        self.enc6 = DownsamplerEncoder(256, 256)

        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv2d(128, 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv2d(64, 21, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(21)
        )

        self.upsample1  = nn.UpsamplingBilinear2d(scale_factor = 4)
        self.conv5      = nn.Sequential(
            DepthwiseSeparableConv2d(21, 21, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(21)
        )

        self.upsample2  = nn.UpsamplingBilinear2d(scale_factor = 4)
        self.conv6      = nn.Sequential(
            DepthwiseSeparableConv2d(21, 21, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(21)
        )

    def forward(self, x):
        x   = self.conv1(x)
        x   = self.conv2(x)

        x   = self.enc1(x)
        x   = self.enc2(x)
        x   = self.enc3(x)
        x   = self.enc4(x)
        x   = self.enc5(x)
        x   = self.enc6(x)

        x   = self.conv3(x)
        x   = self.conv4(x)

        x   = self.upsample1(x)
        x1  = self.conv5(x)
        x1  = x + x1

        x1  = self.upsample2(x1)
        x2  = self.conv6(x1)
        x2  = x1 + x2

        return x2

class PennFudanDataset(object):
    def __init__(self, root, transforms = None):
        self.root = root
        self.transforms = transforms

        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs   = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks  = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path    = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path   = os.path.join(self.root, "PedMasks", self.masks[idx])
        img         = Image.open(img_path).convert("RGB")

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask    = Image.open(mask_path)

        # convert the PIL Image into a numpy array
        mask    = np.array(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)

        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks   = mask == obj_ids[:, None, None]

        # there is only one class
        masks   = torch.as_tensor(masks, dtype = torch.uint8)

        if self.transforms is not None:
            img, masks = self.transforms(img, masks)

        return img, masks

    def __len__(self):
        return len(self.imgs)

dataset     = PennFudanDataset('PennFudanPed')
trainloader = torch.utils.data.DataLoader(dataset, batch_size = 2, shuffle = True, num_workers = 4)

net         = Deeplabv4()
criterion   = nn.CrossEntropyLoss()
optimizer   = optim.SGD(net.parameters())

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')