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
        
        

class MSA(nn.Module):
    
    def __init__(self, embedding_dim: int=768, num_heads: int=12, dropout: float=0):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        self.msa_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                   num_heads=num_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        
        def forward(self, x):
            x = self.layer_norm(x)
            x, _ = self.msa_attention(query=x,
                                      key=x,
                                      value=x,
                                      need_weights=False)
            return x
        
        
class MLP(nn.Module):
    def __init__(self, 
                 embedding_dim: int=768,
                 mlp_size: int=3072,
                 dropout: float=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        self.mlp = nn.Sequential(
                nn.Linear(in_features=embedding_dim,
                        out_features=mlp_size),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=mlp_size,
                        out_features=embedding_dim),
                nn.Dropout(p=dropout))
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        
        return x
        
        
        