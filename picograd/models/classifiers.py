"""
picograd/models/mlp.py  — DenseClassifier (port of glob_train.py model)
picograd/models/lenet.py — LeNet-5
"""

import numpy as np
import picograd.nn as nn
from picograd.nn.module import Module
from picograd.tensor import Tensor


class DenseClassifier(Module):
    """
    Direct port of the existing glob_train.py model:
    Linear(784,512)→ReLU→Dropout(0.2)→Linear(512,256)→ReLU→Dropout(0.2)→Linear(256,10)
    """
    def __init__(self, in_features=784, hidden1=512, hidden2=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return self.fc3(x)


class LeNet5(Module):
    """
    Classic LeNet-5 for 28×28 grayscale input (10 classes).
    Conv→ReLU→MaxPool→Conv→ReLU→MaxPool→FC→FC→FC
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1  = nn.Conv2d(in_channels, 6, kernel_size=5, padding=2)
        self.relu1  = nn.ReLU()
        self.pool1  = nn.MaxPool2d(2)
        self.conv2  = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2  = nn.ReLU()
        self.pool2  = nn.MaxPool2d(2)
        self.flat   = nn.Flatten()
        self.fc1    = nn.Linear(16 * 5 * 5, 120)
        self.relu3  = nn.ReLU()
        self.fc2    = nn.Linear(120, 84)
        self.relu4  = nn.ReLU()
        self.fc3    = nn.Linear(84, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flat(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        return self.fc3(x)


__all__ = ["DenseClassifier", "LeNet5"]
