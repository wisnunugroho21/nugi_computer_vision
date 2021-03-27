import torch
import torch.nn as nn
import torchvision.transforms as transforms

from dataloader.DogBreedDataset import DogBreedDataset
from loss.simclr import SimCLR
from model.image_semantic.encoder import Encoder
from model.image_semantic.decoder import Decoder
from model.clr.projection import Projection

PATH = '.'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trans0 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trans1 = transforms.Compose([
    transforms.RandomResizedCrop(320),                           
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.8),
    transforms.RandomGrayscale(p = 0.2),
    transforms.GaussianBlur(33),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trans2 = transforms.Compose([    
    transforms.RandomResizedCrop(320),                           
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.8),
    transforms.RandomGrayscale(p = 0.2),
    transforms.GaussianBlur(33),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset    = DogBreedDataset(root = '', transforms = trans0)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 8, shuffle = False, num_workers = 2)

clrset1     = DogBreedDataset(root = '', transform = trans1)
clrloader1  = torch.utils.data.DataLoader(clrset1, batch_size = 8, shuffle = False, num_workers = 2)

clrset2     = DogBreedDataset(root = '', transform = trans2)
clrloader2  = torch.utils.data.DataLoader(clrset2, batch_size = 8, shuffle = False, num_workers = 2)

testset     = DogBreedDataset(root = '', transform = trans0)
testloader  = torch.utils.data.DataLoader(testset, batch_size = 8, shuffle = False, num_workers = 2)

encoder     = Encoder()
projector   = Projection()

encoder, projector = encoder.to(device), projector.to(device)

clroptimizer    = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr = 0.001)
clrloss         = SimCLR()

for epoch in range(20):
    running_loss = 0.0
    for data1, data2 in zip(clrloader1, clrloader2):
        input1, _   = data1        
        input2, _   = data2

        input1, input2  = input1.to(device), input2.to(device)

        clroptimizer.zero_grad()

        mid1   = encoder(input1)
        out1   = projector(mid1)

        mid2   = encoder(input2)
        out2   = projector(mid2)

        loss = clrloss.compute_loss(out1, out2)
        loss.backward()
        clroptimizer.step()

    print('loop clr -> ', epoch)

print('Finished Pre-Training')
torch.save(encoder.state_dict(), PATH + '/encoder.pth')

decoder = Decoder()
decoder = decoder.to(device)

segoptimizer    = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = 0.001)
segloss         = nn.CrossEntropyLoss()

for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        input, label    = data
        input, label    = input.to(device), label.to(device)

        segoptimizer.zero_grad()

        mid = encoder(input)
        out = decoder(mid)

        loss = segloss(out, label)
        loss.backward()
        segoptimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

torch.save(encoder.state_dict(), PATH + '/encoder.pth')
torch.save(encoder.state_dict(), PATH + '/decoder.pth')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        mid         = encoder(input)
        out         = decoder(mid)

        total   += labels.size(0)
        correct += (out == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
