import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        """
        :param num_layers: number of layers in the architecture (ResNet)
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param identity_downsample: Conv layer to downsample image in case of different input and output channels
        :param kernel: kernel size
        :param stride: stride
        """
        assert num_layers in [50, 101], 'unknown architecture'

        super().__init__()
        self.identity_downsample = identity_downsample

        # every block in ResNet50 or deeper increases the number of in_channels by 4
        self.expansion = 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU()
        )
    
    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # if the input and output channels are different, then downsample (with no activation, hence identity) the input image
        if self.identity_downsample:
            identity = self.identity_downsample(identity)
        
        # add the identity (input image) to the output of the block
        x += identity
        x = F.relu(x)
        return x


class MyResNet(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels):
        """
        :param num_layers: number of layers in the architecture (ResNet)
        :param in_channels: number of input image channels
        :param out_channels: number of output classes
        """
        assert num_layers in [50, 101], 'unknown architecture'

        super().__init__()

        # how many times to reuse the same block in the architecture
        if num_layers == 50:
            self.layers = [3, 4, 6, 3]
        elif num_layers == 101:
            self.layers = [3, 4, 23, 3]
        else:
            raise NotImplementedError('unknown architecture')
        
        # according to the paper, the first layer is 7x7 conv with stride 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
