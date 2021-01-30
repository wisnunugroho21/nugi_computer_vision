import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import torchvision.transforms as transforms

from dataset.ImageDataset import ImageDataset
from model.deeplabv4 import Deeplabv4

device      = torch.device('cuda:0')
PATH        = 'weights/deeplabv4_net.pth'

transform1  = transforms.Compose([
    transforms.Resize((128, 128))
])
transform2  = transforms.Compose([
    transforms.Resize((128, 128))
])

dataset     = ImageDataset('dataset', transform1, transform2)
trainloader = data.DataLoader(dataset, batch_size = 16, shuffle = True, num_workers = 1)

net         = Deeplabv4(num_classes = 3).to(device)
net.train()

criterion   = nn.CrossEntropyLoss()
optimizer   = optim.Adam(net.parameters(), lr = 3e-4)
scaler      = torch.cuda.amp.GradScaler()

for epoch in range(25):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = net(inputs)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(), PATH)