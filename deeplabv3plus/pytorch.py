import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, stride = 1, padding = 0, dilation = 1, bias = False, depth_multiplier = 1, activate_first = True):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(nin, nin * depth_multiplier, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias, groups = nin),
            nn.BatchNorm2d(nin * depth_multiplier)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(nin * depth_multiplier, nout, kernel_size = 1, bias = bias),
            nn.BatchNorm2d(nout)
        )

        self.relu0 = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU() 

    def forward(self, x):
        if self.activate_first:
            x = self.relu0(x)

        x = self.depthwise(x)

        if not self.activate_first:
            x = self.relu1(x)

        x = self.pointwise(x)

        if not self.activate_first:
            x = self.relu2(x)

        return x

class XceptionBlock(nn.Module):
    def __init__(self, in_filters, out_filters, strides = 1, atrous = 1, grow_first = True, activate_first = True):
        super(XceptionBlock, self).__init__()

        atrous          = [atrous] * 3 if isinstance(atrous, int) else atrous
        filters         = out_filters if grow_first else in_filters

        self.sepconv1 = DepthwiseSeparableConv2d(in_filters, filters, kernel_size = 3, stride = 1, padding = 1 * atrous[0], dilation = atrous[0], bias = False, activate_first = activate_first)
        self.sepconv2 = DepthwiseSeparableConv2d(filters, out_filters, kernel_size = 3, stride = 1, padding = 1 * atrous[1], dilation = atrous[1], bias = False, activate_first = activate_first)
        self.sepconv3 = DepthwiseSeparableConv2d(out_filters, out_filters, kernel_size = 3, stride = strides, padding = 1 * atrous[2], dilation = atrous[2], bias = False, activate_first = activate_first)

    def forward(self, inp):
        x1 = self.sepconv1(inp)
        x2 = self.sepconv2(x1)
        x3 = self.sepconv3(x2)

        return x3 + inp

class PoolResXceptionBlock(nn.Module):
    def __init__(self, in_filters, out_filters, strides = 1, atrous = 1, grow_first = True, activate_first = True):
        super(PoolResXceptionBlock, self).__init__()

        atrous          = [atrous] * 3 if isinstance(atrous, int) else atrous
        filters         = out_filters if grow_first else in_filters

        self.sepconv1   = DepthwiseSeparableConv2d(in_filters, filters, kernel_size = 3, stride = 1, padding = 1 * atrous[0], dilation = atrous[0], bias = False, activate_first = activate_first)
        self.sepconv2   = DepthwiseSeparableConv2d(filters, out_filters, kernel_size = 3, stride = 1, padding = 1 * atrous[1], dilation = atrous[1], bias = False, activate_first = activate_first)
        self.sepconv3   = DepthwiseSeparableConv2d(out_filters, out_filters, kernel_size = 3, stride = strides, padding = 1 * atrous[2], dilation = atrous[2], bias = False, activate_first = activate_first)

        self.skip       = nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size = 1, stride = strides, bias=False),
            nn.BatchNorm2d(out_filters, momentum = 0.0003)
        )
    
    def forward(self, inp):        
        skip = self.skip(inp)

        x1 = self.sepconv1(inp)
        x2 = self.sepconv2(x1)
        x3 = self.sepconv3(x2)

        return x3 + skip, x2

class Xception(nn.Module):
    def __init__(self, os):
        super(Xception, self).__init__()

        self.early_conv = nn.Sequential( 
            nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(32, momentum = 0.0003),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(64, momentum = 0.0003),
            nn.ReLU()
        )

        self.block1 = PoolResXceptionBlock(64, 128, strides = 2)
        self.block2 = PoolResXceptionBlock(128, 256, strides = 2)
        self.block3 = PoolResXceptionBlock(256, 728, strides = 2)

        self.middle_block = nn.Sequential(
            XceptionBlock(728, 728, strides = 1, atrous = 1),
            XceptionBlock(728, 728, strides = 1, atrous = 1),
            XceptionBlock(728, 728, strides = 1, atrous = 1),
            XceptionBlock(728, 728, strides = 1, atrous = 1),

            XceptionBlock(728, 728, strides = 1, atrous = 1),
            XceptionBlock(728, 728, strides = 1, atrous = 1),
            XceptionBlock(728, 728, strides = 1, atrous = 1),
            XceptionBlock(728, 728, strides = 1, atrous = 1),

            XceptionBlock(728, 728, strides = 1, atrous = 1),
            XceptionBlock(728, 728, strides = 1, atrous = 1),
            XceptionBlock(728, 728, strides = 1, atrous = 1),
            XceptionBlock(728, 728, strides = 1, atrous = 1),

            XceptionBlock(728, 728, strides = 1, atrous = 1),
            XceptionBlock(728, 728, strides = 1, atrous = 1),
            XceptionBlock(728, 728, strides = 1, atrous = 1),
            XceptionBlock(728, 728, strides = 1, atrous = 1),
        )
        
        self.block4 = PoolResXceptionBlock(728, 1024, strides = 2, atrous = 1, grow_first=False)

        self.end_conv = nn.Sequential(
            DepthwiseSeparableConv2d(1024, 1536, kernel_size = 3, stride = 1, padding = 1, dilation = 1, activate_first = False),
            DepthwiseSeparableConv2d(1536, 1536, kernel_size = 3, stride = 1, padding = 1, dilation = 1, activate_first = False),
            DepthwiseSeparableConv2d(1536, 2048, kernel_size = 3, stride = 1, padding = 1, dilation = 1, activate_first = False)
        )

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def forward(self, input):
        x           = self.early_conv(input)
        
        x, _        = self.block1(x)
        x, out1     = self.block2(x)
        x, _        = self.block3(x)

        x           = self.middle_block(x)
        x, _        = self.block4(x)
        out_final   = self.end_conv(x)

        return out1, out_final

class ASPP(nn.Module):	
	def __init__(self, dim_in, dim_out, rate = 1, bn_mom = 0.1):
		super(ASPP, self).__init__()

		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, kernel_size = 1, stride = 1, padding = 0, dilation = rate, bias = True),
				nn.BatchNorm2d(dim_out, momentum = bn_mom),
				nn.ReLU(),
		)
        
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 6 * rate, dilation = 6 * rate, bias = True),
				nn.BatchNorm2d(dim_out, momentum = bn_mom),
				nn.ReLU(),
		)

		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding = 12 * rate, dilation = 12 * rate, bias = True),
				nn.BatchNorm2d(dim_out, momentum = bn_mom),
				nn.ReLU(),	
		)

		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, kernel_size = 3, stride = 1, padding=18 * rate, dilation = 18 * rate, bias = True),
				nn.BatchNorm2d(dim_out, momentum = bn_mom),
				nn.ReLU(),	
		)

        self.branch5 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(dim_out, momentum = bn_mom),
            nn.ReLU()
        )
		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding = 0, bias = True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(),		
		)

#		self.conv_cat = nn.Sequential(
#				nn.Conv2d(dim_out*4, dim_out, 1, 1, padding=0),
#				SynchronizedBatchNorm2d(dim_out),
#				nn.ReLU(inplace=True),		
#		)
	def forward(self, x):
		[b, c, row, col] = x.size()

		conv1x1     = self.branch1(x)
		conv3x3_1   = self.branch2(x)
		conv3x3_2   = self.branch3(x)
		conv3x3_3   = self.branch4(x)

		global_feature  = torch.mean(x, dim = 2, keepdim = True)
		global_feature  = torch.mean(global_feature, dim = 3, keepdim = True)
		global_feature  = self.branch5(global_feature)
		global_feature  = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim = 1)
#		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class Deeplabv3plus(nn.Module):
	def __init__(self, cfg):
		super(Deeplabv3plus, self).__init__()

        self.aspp = nn.Sequential(
            ASPP(dim_in = 2048, dim_out = 256, rate = 1, bn_mom = 0.0003),
            nn.Dropout(0.5)
        )

		self.upsample_result    = nn.UpsamplingBilinear2d(scale_factor = 4)
		self.upsample_aspp      = nn.UpsamplingBilinear2d(scale_factor = 4)

		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(256, 48, kernel_size = 1, 1, padding = 0,bias = True),
				nn.BatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum = 0.0003),
				nn.ReLU(),		
		)		

		self.cat_conv = nn.Sequential(
				nn.Conv2d(256 + 48, 256, 3, 1, padding = 1,bias = True),
				nn.BatchNorm2d(256, momentum = 0.0003),
				nn.ReLU(),
				nn.Dropout(0.5),
				nn.Conv2d(256, 256, 3, 1, padding=1,bias=True),
				nn.BatchNorm2d(256, momentum = 0.0003),
				nn.ReLU(),
				nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(256, 21, 1, 1, padding=0)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		self.backbone = Xception()

	def forward(self, x): 
		out1, out_final = self.backbone(x)        

		feature_aspp    = self.aspp(out_final)
		feature_aspp    = self.upsample_aspp(feature_aspp)

        feature_shallow = self.shortcut_conv(out1)	
		feature_cat     = torch.cat([feature_aspp, feature_shallow], 1)

		result          = self.cat_conv(feature_cat) 
		result          = self.cls_conv(result)
		result          = self.upsample_result(result)

		return result

import os
import numpy as np
import torch
from PIL import Image

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        # convert the PIL Image into a numpy array
        mask = np.array(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)

        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

from torchvision import train_one_epoch, evaluate
import utils

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")