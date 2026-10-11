import csv
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.config import load
from src.data.preprocess import load_image, simulate_nir
from src.models.build import build_model
from src.utils.io import resolve_path, exists
from src.utils.metrics import classification_metrics, save_metrics


class MultiSpectralDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.images = data["images"].float()
        self.labels = data["labels"].long()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class ManifestDataset(Dataset):
    def __init__(self, csv_path, image_size):
        self.rows = []
        self.image_size = image_size
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        rgb = load_image(row["rgb"], self.image_size, "RGB")
        if "label_class" in row and row["label_class"].strip() != "":
            label = int(row["label_class"])
            if "nir" in row and row["nir"].strip() != "":
                nir = load_image(row["nir"], self.image_size, "L")
                nir = nir[:, :, None]
            else:
                nir = simulate_nir(rgb)
        else:
            nir = load_image(row["nir"], self.image_size, "L")
            nir = nir[:, :, None]
            label_mask = load_image(row["label"], self.image_size, "L")
            label = 1 if label_mask.mean() > 0.02 else 0

        x = torch.from_numpy(
            (np.concatenate([rgb, nir], axis=2)).transpose(2, 0, 1)
        ).float()
        return x, torch.tensor(label, dtype=torch.long)


def evaluate():
    cfg = load()
    val_path = resolve_path("data/processed/val.pt")
    val_csv = resolve_path("data/processed/val.csv")
    model_path = resolve_path("data/processed/model.pt")

    if exists(val_csv):
        ds = ManifestDataset(val_csv, cfg["data"]["image_size"])
    else:
        ds = MultiSpectralDataset(val_path)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"])

    model = build_model(cfg)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(preds.tolist())
            y_prob.append(torch.softmax(logits, dim=1).numpy())

    metrics = classification_metrics(y_true, y_pred, np.concatenate(y_prob))
    save_metrics(metrics, resolve_path("data/processed/metrics.json"))


if __name__ == "__main__":
    evaluate()
