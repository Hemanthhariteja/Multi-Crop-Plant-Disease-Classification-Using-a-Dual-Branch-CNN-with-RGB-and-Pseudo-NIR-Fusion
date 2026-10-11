import os
from pathlib import Path
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def project_root():
    return Path(__file__).resolve().parents[2]


def resolve_path(rel_path):
    return project_root() / rel_path


def exists(path):
    return os.path.exists(path)
