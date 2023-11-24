from torchinfo import summary
import torch
from models.vit import ViT_Ti_32
from models.cvt import CvT, Tokenizer


inp = torch.randn(1, 3, 224, 224)

vit = ViT_Ti_32()

# print(vit(inp))

# summary(model=vit,
#         input_size=(32, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )


# tok = Tokenizer(kernel_size=7,stride=2,padding=3,n_conv_layers=2, in_planes=64, n_output_channels=64)

# print(tok(inp).shape)
# print(tok.sequence_length())

""" config :

kernel_size=7, stride=2, padding=3, n_convs=2, in_plane=64, out_channels=64             1, 196, 64

"""

cvt = CvT()

print(cvt(inp).shape)
summary(model=cvt,
        input_size=(32, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
