import torch
import torch.nn as nn
from basic_layer import BasicConv2d, BasicFC
from inception import Inception, AuxiliaryClassifier


class GoogLeNet(nn.Module):
    def __init__(self, aux = True, num_classes=1000):
        super().__init__()
        self.sequential1 = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        )

        self.sequential2 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.sequential3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.sequential4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.sequential5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.7, inplace=True)
        self.fc = BasicFC(1024, num_classes)

        self.aux1 = AuxiliaryClassifier(512, num_classes) if aux else None
        self.aux2 = AuxiliaryClassifier(528, num_classes) if aux else None

    def forward(self, x):
        x = self.sequential1(x)
        x = self.sequential2(x)
        x = self.sequential3(x)
        x = self.sequential4(x)
        x = self.sequential5(x)
        x = self.fc(self.dropout(self.flatten(x)))
        return x