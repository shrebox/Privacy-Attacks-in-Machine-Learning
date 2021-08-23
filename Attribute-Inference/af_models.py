# Based on https://gist.github.com/erykml/cf4e23cf3ab8897b287754dcb11e2f84#file-lenet_network-py

import torch.nn as nn
import torch

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class AttackModel(nn.Module):
    def __init__(self, dimension):
        super().__init__()

        self.dimension = dimension

        self.classifier = nn.Sequential(
            nn.Linear(dimension, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class TargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        y = self.output(x)
        return y, x
