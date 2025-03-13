import torch
import torch.nn as nn
from basic_layer import BasicConv2d, BasicFC



class GoogLeNet(nn.Module):
    def __init__(
            self,
            ):
        super().__init__()
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return x