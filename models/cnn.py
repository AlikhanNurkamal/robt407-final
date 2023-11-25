import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom CNN architecture (VGG16-like)
class customCNN(nn.Module):
    def __init__(self, in_channels=3, out_classes=10):
        """
        :param in_channels: number of input image channels
        :param out_classes: number of output classes
        """
        super().__init__()

        # the size of the feature maps will be 112x112x64 after the first conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),  # (224 - 3 + 2*1) / 1 + 1 = 224
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (224 - 2) / 2 + 1 = 112
        )
        # the size of the feature maps will be 56x56x128 after the second conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (112 - 3 + 2*1) / 1 + 1 = 112
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (112 - 2) / 2 + 1 = 56
        )
        # the size of the feature maps will be 28x28x256 after the third conv layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (56 - 3 + 2*1) / 1 + 1 = 56
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (56 - 2) / 2 + 1 = 28
        )
        # the size of the feature maps will be 14x14x512 after the fourth conv layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # (28 - 3 + 2*1) / 1 + 1 = 28
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (28 - 2) / 2 + 1 = 14
        )
        # the size of the feature maps will be 7x7x512 after the fifth conv layer
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # (14 - 3 + 2*1) / 1 + 1 = 14
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (14 - 2) / 2 + 1 = 7
        )
        self.fc1 = nn.Linear(7 * 7 * 512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, out_classes)
    
    def forward(self, x):
        x = self.conv1(x)  # (batch_size, 64, 112, 112)
        x = self.conv2(x)  # (batch_size, 128, 56, 56)
        x = self.conv3(x)  # (batch_size, 256, 28, 28)
        x = self.conv4(x)  # (batch_size, 512, 14, 14)
        x = self.conv5(x)  # (batch_size, 512, 7, 7)
        
        x = x.view(x.shape[0], -1)  # (batch_size, 512 * 7 * 7)
        x = F.relu(self.fc1(x))  # (batch_size, 512)
        x = F.relu(self.fc2(x))  # (batch_size, 256)
        x = self.out(x)

        return x


# Block for ResNet50 and ResNet101 architectures
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param identity_downsample: Conv layer to downsample image in case of different input and output channels
        :param stride: stride
        """
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
        x = x + identity
        x = F.relu(x)
        return x

# ResNet50 and ResNet101 architectures
class MyResNet(nn.Module):
    def __init__(self, num_layers, in_channels, out_classes):
        """
        :param num_layers: number of layers in the architecture (ResNet)
        :param in_channels: number of input image channels
        :param out_classes: number of output classes
        """
        assert num_layers in [50, 101], 'unknown architecture'

        super().__init__()

        # how many times to reuse the same block in the architecture
        if num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            raise NotImplementedError('unknown architecture')
        
        self.in_channels = 64

        # according to the paper, the first layer is 7x7 conv with stride 2 and padding 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(layers[0], 64, stride=1)
        self.layer2 = self._make_layer(layers[1], 128, stride=2)
        self.layer3 = self._make_layer(layers[2], 256, stride=2)
        self.layer4 = self._make_layer(layers[3], 512, stride=2)

        # according to the paper, the last layer is avgpool with output size 1x1
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * 4, out_classes)
    
    def _make_layer(self, num_residual_blocks, in_channels, stride):
        """
        :param num_residual_blocks: how many times to reuse the same block in the architecture
        :param in_channels: number of input channels, output channels are 4 times larger
        :param stride: stride
        :return: layers of residual blocks
        """
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != in_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, in_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(in_channels * 4)
            )
        
        # perform the first residual block
        layers.append(Block(self.in_channels, in_channels, identity_downsample, stride))
        self.in_channels = in_channels * 4

        # perform the rest of the residual blocks
        for i in range(num_residual_blocks - 1):
            layers.append(Block(self.in_channels, in_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # first 7x7 conv layer
        x = self.conv1(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # last avgpool layer plus fully connected layer
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


# in this project we have 10 classes to predict, so out_classes=10
def ResNet50(in_channels=3, out_classes=10):
    return MyResNet(50, in_channels, out_classes=out_classes)


# in this project we have 10 classes to predict, so out_classes=10
def ResNet101(in_channels=3, out_classes=10):
    return MyResNet(101, in_channels, out_classes=out_classes)
