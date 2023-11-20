import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from glob import glob


class CNNCustomDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform

        self.images_paths = []
        self.labels = []
        
        # iterate through each folder (name of a folder corresponds to a class)
        for label in list(glob(os.path.join(images_dir, '*'))):
            images = list(glob(os.path.join(images_dir, label, '*.jpg')))
            self.labels.extend([int(label[-1]) for i in range(len(images))])
            self.images_paths.extend([image for image in images])

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        # getting the path to one image
        img_path = self.images_paths[index]

        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


class CNNInferenceDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform

        self.images = os.listdir(images_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # to submit to kaggle competition I need to return image name
        return img_name, image


# logloss function for State Farm Distracted Driver Detection competition
# class logloss(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, labels, preds):
#         labels = labels.float()
#         preds = preds.float()
#         preds = torch.max(torch.min(preds, torch.tensor(1 - 1e-15)), torch.tensor(1e-15))
#         loss = -torch.mean(torch.sum(labels * torch.log(preds), dim=1))
#         return loss.item()
