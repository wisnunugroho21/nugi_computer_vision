import torch.nn as nn
from model.component.depthwise_separable_conv2d import DepthwiseSeparableConv2d

class Decoder(nn.Module):
    def __init__(self, num_classes = 3):
        super(Decoder, self).__init__()       

        self.back_channel_extractor = nn.Sequential(
            DepthwiseSeparableConv2d(256, num_classes, kernel_size = 3, stride = 1, padding = 1),
            nn.ELU(),
        )

        self.upsample1  = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1),
            nn.ELU(),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1),
            nn.ELU()
        )

        self.upsample2  = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1),
            nn.ELU(),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size = 4, stride = 2, padding = 1),
            nn.ELU()
        )

    def forward(self, x):
        x   = self.back_channel_extractor(x)

        x   = self.upsample1(x)        
        x   = self.upsample2(x)

        return x