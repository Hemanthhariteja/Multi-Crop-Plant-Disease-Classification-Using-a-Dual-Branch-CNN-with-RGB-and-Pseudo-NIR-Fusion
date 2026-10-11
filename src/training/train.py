import csv
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from src.config import load
from src.data.preprocess import ensure_processed_data, load_image, simulate_nir
from src.models.build import build_model
from src.models.losses import get_classification_loss
from src.training.callbacks import EarlyStopping
from src.utils.io import resolve_path, exists
from src.utils.logger import get_logger


logger = get_logger("train")


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


def get_device(cfg):
    if cfg["train"]["device"] == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg["train"]["device"]


def encode_dataset(model, dataset, batch_size, device, flip=False):
    # The backbone is frozen, so run it once and train the fusion head on cached features.
    model.eval()
    feats, labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size)
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        feats.append(model.encode(x).cpu())
        labels.append(y)
        if flip:
            feats.append(model.encode(torch.flip(x, dims=[3])).cpu())
            labels.append(y)
        if (i + 1) % 50 == 0:
            logger.info("encoded %d/%d batches", i + 1, len(loader))
    return TensorDataset(torch.cat(feats), torch.cat(labels))


def train():
    cfg = load()
    ensure_processed_data(cfg, augment=True)

    train_path = resolve_path("data/processed/train.pt")
    val_path = resolve_path("data/processed/val.pt")
    train_csv = resolve_path("data/processed/train.csv")
    val_csv = resolve_path("data/processed/val.csv")

    if exists(train_csv) and exists(val_csv):
        train_ds = ManifestDataset(train_csv, cfg["data"]["image_size"])
        val_ds = ManifestDataset(val_csv, cfg["data"]["image_size"])
    else:
        train_ds = MultiSpectralDataset(train_path)
        val_ds = MultiSpectralDataset(val_path)

    device = get_device(cfg)
    model = build_model(cfg, pretrained=True).to(device)
    batch_size = cfg["train"]["batch_size"]

    # net is the part that trains: the whole model, or just the head over cached features.
    net = model
    if hasattr(model, "encode"):
        logger.info("encoding %d train / %d val images with the frozen backbone", len(train_ds), len(val_ds))
        train_ds = encode_dataset(model, train_ds, batch_size, device, flip=True)
        val_ds = encode_dataset(model, val_ds, batch_size, device)
        net = model.fusion

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    loss_fn = get_classification_loss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    stopper = EarlyStopping(patience=cfg["train"].get("patience", 3))

    best_val = None
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = net(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = net(x)
                val_loss += loss_fn(logits, y).item()

        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f",
            epoch + 1,
            train_loss / len(train_loader),
            val_loss / len(val_loader),
        )

        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), resolve_path("data/processed/model.pt"))

        if stopper.step(val_loss):
            logger.info("early stopping triggered")
            break


if __name__ == "__main__":
    train()
