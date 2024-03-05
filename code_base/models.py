import torch
import torch.nn as nn
import torch.nn.functional as F


class TestNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 8, 3)
        self.fc1 = nn.Linear(8 * 10 * 13, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 15)

    def forward(self, x):
        # Input images of size 1x48x60
        x = F.relu(self.conv1(x))  # 3x46x58
        x = self.max_pool(x)  # 3x23x29
        x = F.relu(self.conv2(x))  # 8x21x27
        x = self.max_pool(x)  # 8x10x13
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
