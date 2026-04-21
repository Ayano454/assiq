from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet50


class CIFARResNet50(nn.Module):
    """ResNet50 adapted for CIFAR-10 (32x32 images)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.model = resnet50(weights=None)
        # CIFAR-10 friendly stem.
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)



def build_model(num_classes: int = 10) -> nn.Module:
    return CIFARResNet50(num_classes=num_classes)
