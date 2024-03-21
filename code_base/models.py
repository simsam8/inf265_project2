import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalNet1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 8, 3)
        self.fc1 = nn.Linear(8 * 10 * 13, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 15)

    def forward(self, x):
        # Input size of 1x48x60
        x = F.relu(self.conv1(x))  # 3x46x58
        x = self.max_pool(x)  # 3x23x29
        x = F.relu(self.conv2(x))  # 8x21x27
        x = self.max_pool(x)  # 8x10x13
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LocalNet2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.max_pool3 = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 24, 3, padding=1)
        self.conv4 = nn.Conv2d(24, 48, 3, padding=1)
        self.fc1 = nn.Linear(48 * 4 * 5, 960)
        self.fc2 = nn.Linear(960, 320)
        self.fc3 = nn.Linear(320, 80)
        self.fc4 = nn.Linear(80, 15)

    def forward(self, x):
        # Input of size 1x48x60
        x = F.relu(self.conv1(x))  # 6x48x60
        x = self.max_pool2(x)  # 6x24x30
        x = F.relu(self.conv2(x))  # 12x24x30
        x = self.max_pool2(x)  # 12x12x15
        x = F.relu(self.conv3(x))  # 24x12x15
        x = self.max_pool3(x)  # 24x4x5
        x = F.relu(self.conv4(x))  # 48x4x5
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DetectNet1_2x3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 6, 3, padding=1)
        self.max_pool_by_3 = nn.MaxPool2d(3, 3)
        self.conv4 = nn.Conv2d(6, 6, 3, padding=0)

    def forward(self, x):
        # Input of size 1x48x60
        x = F.relu(self.conv1(x))  # 2x48x60
        x = self.max_pool(x)  # 2x24x30
        x = F.relu(self.conv2(x))  # 4x24x30
        x = self.max_pool(x)  # 4x12x15
        x = F.relu(self.conv3(x))  # 6x12x15
        x = self.max_pool_by_3(x)  # 6x4x5
        x = self.conv4(x)  # 6x2x3
        return x


class DetectNet2_2x3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.max_pool3 = nn.MaxPool2d(3, 3)
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv4 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv8 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv_out = nn.Conv2d(8, 6, 3, padding=0)

    def forward(self, x):
        # Input of size 1x48x60
        x = F.relu(self.conv1(x))  # 2x48x60
        x = F.relu(self.conv2(x))  # 4x48x60
        x = self.max_pool2(x)  # 4x24x30
        x = F.relu(self.conv3(x))  # 8x24x30
        x = F.relu(self.conv4(x))  # 16x24x30
        x = self.max_pool2(x)  # 16x12x15
        x = F.relu(self.conv5(x))  # 32x12x15
        x = F.relu(self.conv6(x))  # 32x12x15
        x = self.max_pool3(x)  # 32x4x5
        x = F.relu(self.conv7(x))  # 16x4x5
        x = F.relu(self.conv8(x))  # 8x4x5
        x = self.conv_out(x)  # 6x2x3
        return x
