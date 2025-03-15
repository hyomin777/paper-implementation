import torch
import torch.nn as nn
from basic_layer import BasicConv2d, BasicFC


class Inception(nn.Module):
    def __init__(
            self, 
            in_channels, 
            conv_1x1,
            conv_3x3_reduce,
            conv_3x3,
            conv_5x5_reduce,
            conv_5x5,
            pool_proj
            ):
        super().__init__()

        self.branch1 = BasicConv2d(in_channels, conv_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, conv_3x3_reduce, kernel_size=1),
            BasicConv2d(conv_3x3_reduce, conv_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, conv_5x5_reduce, kernel_size=1),
            BasicConv2d(conv_5x5_reduce, conv_5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )


    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = [branch1, branch2, branch3, branch4]

        out = torch.cat(out, 1)
        return out
    

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = BasicFC(128 * 4 * 4, 1024)
        self.fc2 = BasicFC(1024, num_classes)
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x):
        x = self.conv(self.avg_pool(x))
        x = self.fc1(self.flatten(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x