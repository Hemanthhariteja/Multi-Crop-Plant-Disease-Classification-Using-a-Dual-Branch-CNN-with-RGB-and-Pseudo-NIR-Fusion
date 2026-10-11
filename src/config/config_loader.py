from src.utils.io import load_yaml, resolve_path


def load():
    return load_yaml(resolve_path("src/config/config.yaml"))
