import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_features, out_features, kernel=3, stride=1, padding=0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel, stride, padding),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return x


class MyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = Block(in_features, 32, kernel=3, stride=1, padding=0)
        self.conv2 = Block(32, 64, kernel=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
