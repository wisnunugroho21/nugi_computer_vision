import torch
import torch.utils.data as data

import os
import numpy as np
from PIL import Image

class PennFudanPedDataset(data.Dataset):
    def __init__(self, root = '', transforms1 = None, transforms2 = None ):
        self.transforms1    = transforms1
        self.transforms2    = transforms2
        self.root           = root

        self.imgs   = list(sorted(os.listdir(os.path.join(self.root, 'images'))))
        self.masks  = list(sorted(os.listdir(os.path.join(self.root, 'annotations'))))

    def __getitem__(self, idx):        
        img_path    = os.path.join(self.root, 'images', self.imgs[idx])
        mask_path   = os.path.join(self.root, 'annotations', self.masks[idx])
        
        img         = Image.open(img_path).convert("RGB")
        masks       = Image.open(mask_path)

        if self.transforms1 is not None:
            img     = self.transforms1(img)

        if self.transforms2 is not None:    
            masks   = self.transforms2(masks)
        
        img     = torch.FloatTensor(np.array(img))
        img     = img.transpose(1, 2).transpose(0, 1) / 255.0
        
        masks   = torch.LongTensor(np.array(masks))
        masks[masks > 0] = 1.0

        return img, masks

    def __len__(self):
        return len(self.imgs)