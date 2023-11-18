import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob


class TrainDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.labels = []

        for label in list(glob(os.path.join(data_dir, '*'))):
            pass
    
    def __len__(self):
        pass

    def __getitem__(self, index):
        return None, None


class InferenceDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform

        self.images = os.listdir(images_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.images[index])

        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            image = self.transform(image)
        
        return image
