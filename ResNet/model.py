import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = BasicConv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.block1 = ResidualBlock(in_channels=16, out_channels=16, stride=1)
        self.block2 = ResidualBlock(in_channels=16, out_channels=32, stride=2)
        self.block3 = ResidualBlock(in_channels=32, out_channels=64, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)     
        x = self.block1(x) 
        x = self.block2(x)   
        x = self.block3(x)      

        x = F.adaptive_avg_pool2d(x, (1, 1)) 
        x = x.view(x.size(0), -1)        

        x = self.fc(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            BasicConv2d(in_channels, out_channels, stride=stride),
            nn.Conv2d(out_channels, out_channels, stride=1, padding=1),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        identity = x
        out:Tensor = self.conv(x)
        identity = self._shortcut_transform(identity, out.shape)
        return F.relu(out + identity)
    
    def _shortcut_transform(self, identity:Tensor, out_shape):
        _, in_c, in_h, in_w = identity.shape
        _, out_c, out_h, out_w = out_shape

        if in_h > out_h or in_w > out_w:
            identity = self.pool(identity)

        if in_c < out_c:
            pad_c = out_c - in_c
            identity = F.pad(identity, (0,0,0,0,0,pad_c))

        return identity