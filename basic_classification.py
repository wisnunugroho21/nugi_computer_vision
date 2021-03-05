import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomApply

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trans_crop = transforms.Compose([
    transforms.RandomCrop(24),
    transforms.Resize(32)
])

trans_jitter = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    transforms.RandomGrayscale(p=0.2)
])

trainset    = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset     = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader  = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),                       
        )        

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size = 8, stride = 4, padding = 2),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(),
        )      
        
        self.class_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

        self.project_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        x23 = x2 + x3
        x4 = self.conv4(x23)
        x4 = x4.mean([2, 3])
        return self.class_net(x4), self.project_net(x4)

def clrloss(first_encoded, second_encoded):
    first_encoded   = ((first_encoded - first_encoded.mean()) / (first_encoded.std() + 1e-6))
    second_encoded  = ((second_encoded - second_encoded.mean()) / (second_encoded.std() + 1e-6))

    encoded         = torch.cat((first_encoded, second_encoded), dim = 0)
    zeros           = torch.zeros(encoded.shape[0]).long().to(device)    
    
    similarity      = torch.nn.functional.cosine_similarity(encoded.unsqueeze(1), encoded.unsqueeze(0), dim = 2)
    return torch.nn.functional.cross_entropy(similarity, zeros)

net = Net()
net.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

for epoch in range(25):  # loop over the dataset multiple times

    running_loss = 0.0
    for data in trainloader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, _       = data
        crop_inputs     = trans_crop(inputs).to(device)
        jitter_inputs   = trans_jitter(inputs).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        _, crop_outputs     = net(crop_inputs)
        _, jitter_outputs   = net(jitter_inputs)

        loss = clrloss(crop_outputs, jitter_outputs)
        loss.backward()
        optimizer.step()

    print('loop clr -> ', epoch)

for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, _ = net(inputs)
        loss  = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

net = Net()
net.to(device)
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs, _  = net(images)
        predicted   = torch.argmax(outputs, 1)

        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))