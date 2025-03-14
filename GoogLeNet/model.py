from typing import Optional
from collections import namedtuple

import torch.nn as nn
from torch import Tensor

from basic_layer import BasicConv2d, BasicFC
from inception import Inception, AuxiliaryClassifier


GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits1", "aux_logits2"])
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits1": Optional[Tensor], "aux_logits2": Optional[Tensor]}


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux=True):
        super().__init__()
        self.aux = aux
        self.sequential1 = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.sequential2 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.sequential3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Inception(480, 192, 96, 208, 16, 48, 64)
        )

        self.sequential4 = nn.Sequential(
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
        )

        self.sequential5 = nn.Sequential(
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(1024, num_classes)

        self.aux1 = AuxiliaryClassifier(512, num_classes) if aux else None
        self.aux2 = AuxiliaryClassifier(528, num_classes) if aux else None

    def forward(self, x):
        x = self.sequential1(x)
        x = self.sequential2(x)
        x = self.sequential3(x)
        aux1 = self.aux1(x) if self.aux and self.training else None

        x = self.sequential4(x)
        aux2 = self.aux2(x) if self.aux and self.training else None

        x = self.sequential5(x)
        x = self.fc(self.dropout(self.flatten(x)))
        return GoogLeNetOutputs(x, aux1, aux2)