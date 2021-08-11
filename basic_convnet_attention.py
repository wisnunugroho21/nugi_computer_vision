import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

transform_train = transforms.Compose([
    transforms.autoaugment.AutoAugment(torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

batch_size  = 64
epochs      = 100

trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 8)

testset = torchvision.datasets.CIFAR10(root='./data', train = False, download = True, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 8)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

import torch.nn as nn
import torch.nn.functional as F

from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d
from model.cmt.extract_encoder import ExtractEncoder
from model.cmt.downsampler import Downsampler
from model.component.depth_atten_conv import PointAttentConv


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(            
            DepthwiseSeparableConv2d(3, 32, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.GELU(),
            nn.BatchNorm2d(32),
            Downsampler(32, 64),
            ExtractEncoder(64, 8),
            Downsampler(64, 128),
            ExtractEncoder(128, 4),
            Downsampler(128, 256),
            ExtractEncoder(256, 2),
            Downsampler(256, 512),
            ExtractEncoder(512, 1),
            Downsampler(512, 1024),
        )

        self.net = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ELU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.mean([2, 3])

        return self.net(x)

net = Net()
net = net.to(device)

import torch.optim as optim

criterion   = nn.CrossEntropyLoss()
optimizer   = optim.AdamW(net.parameters(), lr = 0.1)
scheduler   = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.01, steps_per_epoch = len(trainloader), epochs = epochs)
scaler      = GradScaler()

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with autocast():
            outputs = net(inputs)
            loss = criterion(outputs, labels)

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
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))