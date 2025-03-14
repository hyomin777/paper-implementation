import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = BasicConv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.block1 = make_block(in_channels=16, out_channels=16, stride=1)
        self.block2 = make_block(in_channels=16, out_channels=32, stride=2)
        self.block3 = make_block(in_channels=32, out_channels=64, stride=2)
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
    

def make_block(in_channels, out_channels, stride):
    layers = []
    layers.append(BasicConv2d(in_channels, out_channels, stride=stride))
    layers.append(BasicConv2d(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)
