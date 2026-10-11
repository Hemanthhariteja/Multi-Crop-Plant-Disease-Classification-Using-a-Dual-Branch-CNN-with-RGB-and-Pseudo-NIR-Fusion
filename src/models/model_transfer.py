import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class TransferMultiBranch(nn.Module):
    """Frozen ImageNet ResNet-18 encodes the RGB and NIR branches; only the fusion head trains."""

    feature_dim = 512

    def __init__(self, num_classes=3, hidden_dim=256, nir_size=64, dropout=0.3, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.nir_size = nir_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def train(self, mode=True):
        super().train(mode)
        # The backbone is frozen, so its batch-norm statistics must stay fixed too.
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode(self, x):
        rgb = (x[:, 0:3, :, :] - self.mean) / self.std
        # NIR is a single band: run it at reduced resolution, repeated to 3 channels.
        nir = F.interpolate(
            x[:, 3:4, :, :], size=(self.nir_size, self.nir_size), mode="bilinear", align_corners=False
        )
        nir = (nir.repeat(1, 3, 1, 1) - self.mean) / self.std
        return torch.cat([self.backbone(rgb), self.backbone(nir)], dim=1)

    def forward(self, x):
        return self.fusion(self.encode(x))
