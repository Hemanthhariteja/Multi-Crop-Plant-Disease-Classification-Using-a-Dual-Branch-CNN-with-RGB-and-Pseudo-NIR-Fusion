from pathlib import Path
from src.utils.io import ensure_dir, resolve_path


def download_placeholder():
    raw_dir = resolve_path("data/raw")
    ensure_dir(raw_dir)
    ensure_dir(Path(raw_dir) / "samples")
    marker = Path(raw_dir) / "README.txt"
    if not marker.exists():
        marker.write_text(
            "Place real data here. Create manifest.csv with one of these formats:\n"
            "1) Multi-spectral + mask: rgb,nir,label (label is a binary mask image)\n"
            "2) RGB-only classification: rgb,label_class (integer class index)\n",
            encoding="utf-8",
        )

    manifest = Path(raw_dir) / "manifest.csv"
    if not manifest.exists():
        manifest.write_text(
            "rgb,label_class,nir,label\n"
            "data/raw/samples/rgb_0001.png,0,data/raw/samples/nir_0001.png,data/raw/samples/label_0001.png\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    download_placeholder()
