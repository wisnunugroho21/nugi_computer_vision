import torch
import torch.utils.data as data

import os
import numpy as np
from PIL import Image

masks   = Image.open('dataset/Pet/annotations/Abyssinian_1.png')
masks   = np.array(masks)

print(masks.shape)

print(np.max(masks))
print(np.min(masks))
print(np.mean(masks))
print(np.std(masks))