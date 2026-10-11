import json
import numpy as np
import torch
from PIL import Image
from src.config import load
from src.models.build import build_model
from src.utils.io import resolve_path, exists


def load_rgb_image(file, image_size):
    # Same resize as training (src.data.preprocess.load_image).
    image = Image.open(file).convert("RGB")
    image = image.resize((image_size, image_size))
    return np.array(image).astype("float32") / 255.0


def load_npy(file):
    arr = np.load(file)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(
            f"Expected an HxWx3 (RGB) or HxWx4 (RGB+NIR) array, got shape {arr.shape}."
        )
    is_int = np.issubdtype(arr.dtype, np.integer)
    arr = arr.astype("float32")
    # Integer arrays (or floats outside 0-1) are assumed to be on a 0-255 scale.
    if is_int or arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def simulate_nir(rgb):
    gray = np.mean(rgb, axis=2, keepdims=True)
    return gray


def to_tensor(rgb, nir, image_size):
    x = np.concatenate([rgb, nir], axis=2).astype("float32")
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0)
    if x.shape[2:] != (image_size, image_size):
        x = torch.nn.functional.interpolate(
            x, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    return x


def predict_from_file(file, is_npy=False):
    cfg = load()
    model = build_model(cfg)
    model.load_state_dict(torch.load(resolve_path("data/processed/model.pt"), map_location="cpu"))
    model.eval()

    image_size = cfg["data"]["image_size"]
    if is_npy:
        arr = load_npy(file)
        rgb = arr[:, :, 0:3]
        if arr.shape[2] >= 4:
            nir = arr[:, :, 3:4]
        else:
            nir = simulate_nir(rgb)
    else:
        rgb = load_rgb_image(file, image_size)
        nir = simulate_nir(rgb)

    x = to_tensor(rgb, nir, image_size)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    class_name = None
    class_map_path = resolve_path("data/raw/class_map.json")
    if exists(class_map_path):
        with open(class_map_path, "r", encoding="utf-8") as f:
            class_map = json.load(f)
        inv_map = {v: k for k, v in class_map.items()}
        class_name = inv_map.get(pred)
    return pred, class_name
