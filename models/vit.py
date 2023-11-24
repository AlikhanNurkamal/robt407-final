import torch
import torch.nn as nn
import torch.nn.functional as F


# convering image into embeddings
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int=3, patch_size: int=16, embedding_dim: int=192):
        super().__init__()
        # patches are extracted using convolution, 
        self.patches = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size, # non overlapping
                                 padding=0)
        # extracted patches are then flattened into embeddings
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        
        
    def forward(self, x):
        x = self.patches(x)
        x = self.flatten(x)
        
        return x.permute(0, 2, 1)
        
        
# MultiHeadAttention Module
class MSA(nn.Module):
    def __init__(self, embedding_dim: int=192, num_heads: int=3, dropout: float=0):
        super().__init__()
        # layer normalization layer
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        # multiheadattention layer
        self.msa_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                   num_heads=num_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        
    def forward(self, x):
        x = self.layer_norm(x)
        # query, key, value are formed from the same x
        x, _ = self.msa_attention(query=x,
                                    key=x,
                                    value=x,
                                    need_weights=False)
        return x
    
# MLP layer   
class MLP(nn.Module):
    def __init__(self, 
                 embedding_dim: int=192,
                 mlp_size: int=768,
                 dropout: float=0.1):
        super().__init__()
        # layer normalization
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        # two linear layers separated by GeLU activation
        self.mlp = nn.Sequential(
                nn.Linear(in_features=embedding_dim,
                        out_features=mlp_size), # from input_dim to mlp_size
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=mlp_size, # from mlp_size to input_dim
                        out_features=embedding_dim),
                nn.Dropout(p=dropout))
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        
        return x
        
# Transformer encoder block
class EncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int=192,
                 num_heads: int=3,
                 mlp_size: int=768,
                 mlp_dropout: float=0.1,
                 msa_dropout: float=0.0):
        super().__init__()
        # each block (layer) consists of MSA and MLP
        self.msa = MSA(embedding_dim=embedding_dim,
                       num_heads=num_heads,
                       dropout=msa_dropout)
        
        self.mlp = MLP(embedding_dim=embedding_dim,
                       mlp_size=mlp_size,
                       dropout=mlp_dropout)
        
    def forward(self, x):
        x = self.msa(x) + x # skip connection #1
        x = self.mlp(x) + x # skip connection #2
        
        return x
    
# ViT class
class ViT(nn.Module):
    def __init__(self,
                 img_size: int=224,
                 in_channels: int=3,
                 patch_size: int=16,
                 layers: int=12,
                 embedding_dim: int=192,
                 mlp_size: int=768,
                 num_heads: int=3,
                 msa_dropout: float=0.0,
                 mlp_dropout: float=0.1,
                 emb_dropout: float=0.1,
                 num_classes: int=1000):
        super().__init__()
        # calculating number of patches
        self.num_patches = (img_size*img_size) // patch_size**2
        # creating additional class embedding (token)
        self.class_embedding = nn.Parameter(data=torch.rand(1, 1, embedding_dim), requires_grad=True)
        # creating learnable positional embeddings
        self.pos_embedding = nn.Parameter(data=torch.rand(1, self.num_patches+1, embedding_dim), requires_grad=True)
        
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        # image -> embeddings
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        # transformer layers
        self.encoder = nn.Sequential(*[
            EncoderBlock(embedding_dim=embedding_dim,
                         num_heads=num_heads,
                         mlp_size=mlp_size,
                         mlp_dropout=mlp_dropout,
                         msa_dropout=msa_dropout)
            for _ in range(layers)
        ])
        # classification head 
        self.head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        # class token
        cls_token = self.class_embedding.expand(batch_size, -1, -1) 
        
        x = self.patch_embedding(x) # embeddings
        x = torch.cat((cls_token, x), dim=1) # embeddings + class token
        x = self.pos_embedding + x # making positional embeddings
        x = self.emb_dropout(x)
        x = self.encoder(x) # transformer layers
        x = self.head(x[:, 0]) # classification is done based on class embedding (token)
        
        return x
        
    