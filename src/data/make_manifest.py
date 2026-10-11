import json
import sys
from pathlib import Path
from src.utils.io import ensure_dir, resolve_path


def find_images(root):
    exts = {".png", ".jpg", ".jpeg"}
    for path in root.rglob("*"):
        if path.suffix.lower() in exts:
            yield path


def build_manifest(dataset_root, out_manifest, out_class_map):
    dataset_root = Path(dataset_root)
    class_names = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])
    class_map = {name: idx for idx, name in enumerate(class_names)}

    rows = ["rgb,label_class"]
    for class_name in class_names:
        class_dir = dataset_root / class_name
        for img_path in find_images(class_dir):
            rows.append(f"{img_path.as_posix()},{class_map[class_name]}")

    Path(out_manifest).write_text("\n".join(rows) + "\n", encoding="utf-8")
    Path(out_class_map).write_text(json.dumps(class_map, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # Optional argument: path to the class_name/xxx.png dataset folder.
    if len(sys.argv) > 1:
        dataset_root = Path(sys.argv[1]).resolve()
    else:
        dataset_root = resolve_path("data/archive (4)/dataset_clean_final")
    ensure_dir(resolve_path("data/raw"))
    out_manifest = resolve_path("data/raw/manifest.csv")
    out_class_map = resolve_path("data/raw/class_map.json")
    build_manifest(dataset_root, out_manifest, out_class_map)
    print("manifest created")
