import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob


class CustomDataset(Dataset):
    def __init__(self, root_dir, imgs, labels, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        # here, imgs and labels are columns of the CSV file
        self.imgs = imgs
        self.labels = labels

        # stacking the root path and the image names
        self.images = [os.path.join(self.root_dir, img) for img in imgs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # getting the path to one image
        img_path = self.images[index]

        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


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
