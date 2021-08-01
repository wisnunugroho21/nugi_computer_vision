import torch
import torch.nn as nn

from model.component.spatial_atrous_extractor import SpatialAtrousExtractor

class AtrousSpatialPyramidConv2d(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out):
        super(AtrousSpatialPyramidConv2d, self).__init__()        

        self.extractor1 = SpatialAtrousExtractor(dim_in, dim_mid, 1)
        self.extractor2 = SpatialAtrousExtractor(dim_in, dim_mid, 4)
        self.extractor3 = SpatialAtrousExtractor(dim_in, dim_mid, 6)
        self.extractor4 = SpatialAtrousExtractor(dim_in, dim_mid, 12)

        self.out = nn.Sequential(
            nn.Conv2d(4 * dim_mid, dim_out, kernel_size = 1),
            nn.ELU()
        )

    def forward(self, x):
        x1 = self.extractor1(x)
        x2 = self.extractor2(x)
        x3 = self.extractor1(x)
        x4 = self.extractor2(x)

        xout = torch.cat((x1, x2, x3, x4), 1)
        xout = self.out(xout)

        return xout