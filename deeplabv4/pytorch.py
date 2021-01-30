import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = rate, dilation = rate, bias = False),            
            nn.ReLU()
		)
        self.conv1  = nn.Sequential(
            nn.BatchNorm2d(dim_in),
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.spatial_atrous(x)

        x1 = self.conv1(x)
        x1 = x + x1

        return x1

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

    def forward(self, x):
        x1 = self.extractor1(x)
        x2 = self.extractor1(x)
        x3 = self.extractor1(x)
        x4 = self.extractor1(x)

        xout = self.bn(x1 + x2 + x3 + x4)
        xout = self.comb_extractor(xout)
        xout = self.change_channel_extractor(xout)

        return xout

class DownsamplerEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DownsamplerEncoder, self).__init__()

        self.downsampler1 = nn.Sequential(            
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.downsampler2 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.ReLU()
        )

        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x1 = self.downsampler1(x)
        x2 = self.downsampler2(x)
        xout   = self.bn(x1 + x2)

        return xout

class ExtractEncoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ExtractEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_in),
            DepthwiseSeparableConv2d(dim_in, dim_in, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),                    
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(dim_in),
            DepthwiseSeparableConv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(dim_out),
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

        self.spatialenc = SpatialEncoder(3, 256)

        self.enc1 = DownsamplerEncoder(256, 256)
        self.enc2 = ExtractEncoder(256, 256)
        self.enc3 = DownsamplerEncoder(256, 256)
        self.enc4 = ExtractEncoder(256, 256)
        self.enc5 = DownsamplerEncoder(256, 256)
        self.enc6 = ExtractEncoder(256, 256)
        self.enc7 = DownsamplerEncoder(256, 256)
        self.enc8 = ExtractEncoder(256, 256)

        self.back_channel_extractor = nn.Sequential(
            DepthwiseSeparableConv2d(256, num_classes, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(3)
        )

        self.upsample1          = nn.UpsamplingBilinear2d(scale_factor = 4)
        self.upsample_conv_1    = nn.Sequential(
            DepthwiseSeparableConv2d(num_classes, num_classes, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )
        self.bn1 = nn.BatchNorm2d(num_classes)

        self.upsample2          = nn.UpsamplingBilinear2d(scale_factor = 4)
        self.upsample_conv_2    = nn.Sequential(
            DepthwiseSeparableConv2d(num_classes, num_classes, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )
        self.bn2 = nn.BatchNorm2d(num_classes)

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
        x1  = self.bn1(x + x1)

        x1  = self.upsample2(x1)
        x2  = self.upsample_conv_2(x1)
        x2  = self.bn2(x1 + x2)

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

def display(img):
    img = torchvision.utils.make_grid(img)
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

transform1 = transforms.Compose([
    transforms.Resize((128, 128))
])

transform2 = transforms.Compose([
    transforms.Resize((128, 128))
])

dataset     = ImageDataset(transform1, transform2)
trainloader = data.DataLoader(dataset, batch_size = 32, shuffle = True, num_workers = 1)

net         = Deeplabv4(num_classes = 3).to(device)
criterion   = nn.CrossEntropyLoss()
optimizer   = optim.Adam(net.parameters(), lr = 3e-4)

scaler      = torch.cuda.amp.GradScaler()

for epoch in range(25):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # Casts operations to mixed precision
        with torch.cuda.amp.autocast():
            outputs = net(inputs)
            loss    = criterion(outputs, labels)

        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

PATH = './deeplabv4_net.pth'
torch.save(net.state_dict(), PATH)