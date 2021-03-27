import torch
import torch.utils.data as data

import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DogBreedDataset(data.Dataset):
    def __init__(self, root = '', transforms = None):
        self.transforms = transforms
        self.root       = root

        self.imgs       = list(sorted(os.listdir(os.path.join(self.root, 'train'))))

        datas           = pd.read_csv(os.path.join(self.root, 'labels.csv'))
        datas['label']  = LabelEncoder().fit_transform(datas.breed)
        self.labels     = torch.tensor(datas['label'].values)

    def __getitem__(self, idx):        
        img_path    = os.path.join(self.root, 'train', self.imgs[idx])        
        img         = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)
        
        # img     = torch.FloatTensor(np.array(img))
        # img     = img.transpose(1, 2).transpose(0, 1) / 255.0

        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.imgs)