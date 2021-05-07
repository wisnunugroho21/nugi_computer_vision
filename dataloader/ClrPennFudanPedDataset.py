import torch
import torch.utils.data as data

import os
import numpy as np
from PIL import Image

class ClrPennFudanPedDataset(data.Dataset):
    def __init__(self, root = '', transforms1 = None, transforms2 = None ):
        self.transforms1    = transforms1
        self.transforms2    = transforms2
        self.root           = root

        self.imgs_anchor    = list(sorted(os.listdir(os.path.join(self.root, 'images'))))
        self.imgs_target    = list(sorted(os.listdir(os.path.join(self.root, 'images'))))

    def __getitem__(self, idx):        
        achor_path    = os.path.join(self.root, 'images', self.imgs_anchor[idx])
        target_path   = os.path.join(self.root, 'images', self.imgs_target[idx])
        
        anchor  = Image.open(achor_path).convert("RGB")
        target  = Image.open(target_path).convert("RGB")

        if self.transforms1 is not None:
            anchor  = self.transforms1(anchor)

        if self.transforms2 is not None:    
            target  = self.transforms2(target)
                
        return anchor, target

    def __len__(self):
        return len(self.imgs_anchor)