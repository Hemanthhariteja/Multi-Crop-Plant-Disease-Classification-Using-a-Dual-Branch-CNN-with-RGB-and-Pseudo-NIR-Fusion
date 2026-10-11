import torch
from torch import nn


class MultiBranchCNN(nn.Module):
    def __init__(self, rgb_channels=3, nir_channels=1, num_classes=3, hidden_dim=64):
        super().__init__()
        self.rgb_branch = nn.Sequential(
            nn.Conv2d(rgb_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.nir_branch = nn.Sequential(
            nn.Conv2d(nir_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        rgb = x[:, 0:3, :, :]
        nir = x[:, 3:4, :, :]
        rgb_f = torch.flatten(self.rgb_branch(rgb), 1)
        nir_f = torch.flatten(self.nir_branch(nir), 1)
        return self.fusion(torch.cat([rgb_f, nir_f], dim=1))
