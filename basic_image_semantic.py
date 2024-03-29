import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from dataloader.PennFudanPedDataset import PennFudanPedDataset
from dataloader.ClrPennFudanPedDataset import ClrPennFudanPedDataset
from dataloader.CatsDataset import CatsDataset

from loss.cmm import ContrastiveMM
from model.image_semantic_segmentation.encoder import Encoder
from model.image_semantic_segmentation.decoder import Decoder
from model.clr.projection import Projection

import matplotlib.pyplot as plt

from copy import deepcopy

def display(display_list, title):
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])

        disImg  = display_list[i].detach().numpy()
        plt.imshow(disImg)
        plt.axis('off')
    plt.show()

epochs = 30
PATH = '.'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""" trans_clr1 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(256),                           
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p = 0.8),
    transforms.RandomGrayscale(p = 0.2),
    transforms.GaussianBlur(25),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trans_clr2 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) """

trans1 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trans2 = transforms.Compose([
    transforms.Resize((128, 128)),
])

""" trans_label  = transforms.Compose([
    transforms.Resize((256, 256))
]) """

""" clrset      = ClrPennFudanPedDataset(root = 'dataset/PennFudanPed', transforms1 = trans_clr1, transforms2 = trans_clr2)
clrloader   = torch.utils.data.DataLoader(clrset, batch_size = 8, shuffle = True, num_workers = 8) """

trainset    = CatsDataset(root = 'dataset/Pet', transforms1 = trans1, transforms2 = trans2)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 8, shuffle = True, num_workers = 8)

testset     = CatsDataset(root = 'dataset/Pet', transforms1 = trans1, transforms2 = trans2)
testloader  = torch.utils.data.DataLoader(testset, batch_size = 8, shuffle = False, num_workers = 8)

encoder     = Encoder()
encoder     = encoder.to(device)

""" encoder, projector      = Encoder(), Projection() 
encoder1, projector1    = deepcopy(encoder), deepcopy(projector)

encoder, projector    = encoder.to(device), projector.to(device)
encoder1, projector1  = encoder1.to(device), projector1.to(device)

clroptimizer    = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr = 0.001)
clrscaler       = torch.cuda.amp.GradScaler()
clrloss         = ContrastiveMM(True)

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(clrloader, 0):
        clroptimizer.zero_grad()

        input1, input2  = data
        input1, input2  = input1.to(device), input2.to(device)

        mid1    = encoder(input1).mean([2, 3])
        out1    = projector(mid1)

        mid2    = encoder1(input2).mean([2, 3])
        out2    = projector1(mid2)

        loss    = clrloss.compute_loss(out1, out2.detach())

        loss.backward()
        clroptimizer.step()        

        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

    print('loop clr -> ', epoch + 1)

print('Finished Pre-Training')
torch.save(encoder.state_dict(), PATH + '/encoder.pth') """

# -----------------------------------------------------------------------------------------------------------

decoder = Decoder(num_classes = 3)
decoder = decoder.to(device)

segoptimizer    = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = 0.001)
segloss         = nn.CrossEntropyLoss()

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        segoptimizer.zero_grad()

        images, labels    = data
        images, labels    = images.to(device), labels.to(device)

        mid = encoder(images)
        out = decoder(mid)

        loss = segloss(out, labels)

        loss.backward()
        segoptimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

# -------------------------------------------------------------------

torch.save(encoder.state_dict(), PATH + '/encoder.pth')
torch.save(decoder.state_dict(), PATH + '/decoder.pth')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        mid = encoder(images)
        out = decoder(mid)

        out = out.transpose(1, 2).transpose(2, 3).argmax(-1)

        total   += (labels.shape[0] * labels.shape[1] * labels.shape[2])
        correct += (out == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

torch.save(encoder.state_dict(), PATH + '/encoder.pth')
torch.save(decoder.state_dict(), PATH + '/decoder.pth')

# -------------------------------------------------------------------

images, labels  = testset[0]
images          = images.unsqueeze(0)
images, labels  = images.to(device), labels.to(device)

mid = encoder(images)
out = decoder(mid)

disInput    = images.squeeze(0).transpose(0, 1).transpose(1, 2)
disOutput   = out.squeeze(0).transpose(0, 1).transpose(1, 2).argmax(-1)

display([disInput.cpu(), labels.cpu(), disOutput.cpu()], ['Input Image', 'True Mask', 'Predicted Mask'])

# -----------------------------------------------------------------------------

images, labels  = testset[10]
images          = images.unsqueeze(0)
images, labels  = images.to(device), labels.to(device)

mid = encoder(images)
out = decoder(mid)

disInput    = images.squeeze(0).transpose(0, 1).transpose(1, 2)
disOutput   = out.squeeze(0).transpose(0, 1).transpose(1, 2).argmax(-1)

display([disInput.cpu(), labels.cpu(), disOutput.cpu()], ['Input Image', 'True Mask', 'Predicted Mask'])

# ----------------------------------------------------------------------------------

images, labels  = testset[14]
images          = images.unsqueeze(0)
images, labels  = images.to(device), labels.to(device)

mid = encoder(images)
out = decoder(mid)

disInput    = images.squeeze(0).transpose(0, 1).transpose(1, 2)
disOutput   = out.squeeze(0).transpose(0, 1).transpose(1, 2).argmax(-1)

display([disInput.cpu(), labels.cpu(), disOutput.cpu()], ['Input Image', 'True Mask', 'Predicted Mask'])