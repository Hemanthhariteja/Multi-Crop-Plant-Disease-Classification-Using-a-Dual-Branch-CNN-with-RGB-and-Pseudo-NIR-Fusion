from pathlib import Path
import csv
import random
import numpy as np
from PIL import Image
import torch
from src.data.augmentations import random_flip
from src.utils.io import ensure_dir, resolve_path, exists
from src.utils.logger import get_logger


logger = get_logger("preprocess")


def generate_synthetic_dataset(image_size, num_classes, count):
    images = torch.rand(count, 4, image_size, image_size)
    labels = torch.randint(0, num_classes, (count,))
    return images, labels


def load_image(path, image_size, mode):
    # Relative manifest paths are relative to the project root, not the cwd.
    image = Image.open(resolve_path(path)).convert(mode)
    image = image.resize((image_size, image_size))
    arr = np.array(image).astype("float32") / 255.0
    return arr


def simulate_nir(rgb):
    gray = rgb.mean(axis=2, keepdims=True)
    return gray


def load_manifest(manifest_path, image_size):
    images = []
    labels = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rgb = load_image(row["rgb"], image_size, "RGB")

            if "label_class" in row and row["label_class"].strip() != "":
                if "nir" in row and row["nir"].strip() != "":
                    nir = load_image(row["nir"], image_size, "L")
                    nir = np.expand_dims(nir, axis=2)
                else:
                    nir = simulate_nir(rgb)
                label = int(row["label_class"])
            else:
                nir = load_image(row["nir"], image_size, "L")
                label_mask = load_image(row["label"], image_size, "L")
                nir = np.expand_dims(nir, axis=2)
                label = 1 if label_mask.mean() > 0.02 else 0

            x = np.concatenate([rgb, nir], axis=2)
            x = np.transpose(x, (2, 0, 1))
            images.append(x)
            labels.append(label)

    images = torch.tensor(np.stack(images), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def split_manifest(manifest_path, train_out, val_out, seed, train_ratio, max_samples=None):
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    random.Random(seed).shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    split_idx = int(len(rows) * train_ratio)
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    for out_path, out_rows in [(train_out, train_rows), (val_out, val_rows)]:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(out_rows)


def ensure_processed_data(config, augment=False):
    processed_dir = resolve_path(config["data"]["processed_dir"])
    ensure_dir(processed_dir)

    train_path = Path(processed_dir) / "train.pt"
    val_path = Path(processed_dir) / "val.pt"
    train_csv = Path(processed_dir) / "train.csv"
    val_csv = Path(processed_dir) / "val.csv"
    manifest_path = resolve_path("data/raw/manifest.csv")

    force_rebuild = config["data"].get("force_rebuild", False)
    if train_path.exists() and val_path.exists() and not force_rebuild:
        if exists(manifest_path):
            manifest_mtime = Path(manifest_path).stat().st_mtime
            train_mtime = train_path.stat().st_mtime
            if manifest_mtime <= train_mtime:
                logger.info("Processed data found.")
                return
        else:
            logger.info("Processed data found.")
            return

    if train_csv.exists() and val_csv.exists() and not force_rebuild:
        if exists(manifest_path):
            manifest_mtime = Path(manifest_path).stat().st_mtime
            train_mtime = train_csv.stat().st_mtime
            if manifest_mtime <= train_mtime:
                logger.info("Processed CSV splits found.")
                return
        else:
            logger.info("Processed CSV splits found.")
            return

    if exists(manifest_path):
        logger.info("Manifest found; creating train/val splits.")
        split_manifest(
            manifest_path,
            train_csv,
            val_csv,
            seed=config["data"].get("split_seed", 42),
            train_ratio=config["data"].get("train_ratio", 0.8),
            max_samples=config["data"].get("max_samples"),
        )
        logger.info("Processed CSV splits saved to processed directory.")
        return
    else:
        logger.info("No manifest found; generating synthetic dataset.")
        image_size = config["data"]["image_size"]
        num_classes = config["data"]["num_classes"]
        train_size = config["data"]["train_size"]
        val_size = config["data"]["val_size"]

        train_images, train_labels = generate_synthetic_dataset(
            image_size, num_classes, train_size
        )
        val_images, val_labels = generate_synthetic_dataset(
            image_size, num_classes, val_size
        )

    if augment:
        train_images = random_flip(train_images)

    torch.save({"images": train_images, "labels": train_labels}, train_path)
    torch.save({"images": val_images, "labels": val_labels}, val_path)
    logger.info("Processed dataset saved to processed directory.")
