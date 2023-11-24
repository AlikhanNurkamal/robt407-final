import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    
    def __init__(self, in_channels: int=3, patch_size: int=16, embedding_dim: int=768):
        super().__init__()
        
        self.patches = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        
        
        def forward(self, x):
            image_resolution = x.shape[-1]
            
            assert image_resolution % patch_size == 0, "Image dimentions must be divisible by patch size"
            
            x = self.patches(x)
            x = self.flatten(x)
            
            return x.permute(0, 2, 1)
        
        
        