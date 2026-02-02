import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage0(nn.Module):
    """
    Early convs: output shape for CIFAR10: (N, 32, 16, 16)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

class Stage1(nn.Module):
    """
    Remaining convs + head: input (N, 32, 16, 16) -> logits (N, C)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def stage_output_shape(in_shape=(3, 32, 32)):
    # for CIFAR10 Stage0 output
    return (32, 16, 16)
