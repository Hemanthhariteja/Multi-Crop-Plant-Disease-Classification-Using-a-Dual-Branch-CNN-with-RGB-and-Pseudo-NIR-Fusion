import json
from src.models.model_cnn import SimpleCNN
from src.models.model_multibranch import MultiBranchCNN
from src.models.model_transfer import TransferMultiBranch
from src.utils.io import resolve_path, exists


def build_model(cfg, pretrained=False):
    # pretrained only matters for training; a saved checkpoint already holds the backbone weights.
    num_classes = cfg["data"]["num_classes"]
    class_map_path = resolve_path("data/raw/class_map.json")
    if exists(class_map_path):
        with open(class_map_path, "r", encoding="utf-8") as f:
            class_map = json.load(f)
        num_classes = len(class_map)
    if cfg["model"]["type"] == "cnn":
        return SimpleCNN(in_channels=4, num_classes=num_classes)
    if cfg["model"]["type"] == "transfer":
        return TransferMultiBranch(
            num_classes=num_classes,
            hidden_dim=cfg["model"]["hidden_dim"],
            pretrained=pretrained,
        )
    return MultiBranchCNN(
        rgb_channels=cfg["model"]["rgb_channels"],
        nir_channels=cfg["model"]["nir_channels"],
        num_classes=num_classes,
        hidden_dim=cfg["model"]["hidden_dim"],
    )
