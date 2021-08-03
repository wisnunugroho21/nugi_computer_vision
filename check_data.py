import torch
import torch.utils.data as data

import os
import numpy as np
from PIL import Image

masks   = Image.open('dataset/PennFudanPed/annotations/FudanPed00001_mask.png')
masks   = np.array(masks)

print(masks.shape)

print(np.max(masks))
print(np.min(masks))
print(np.mean(masks))
print(np.std(masks))