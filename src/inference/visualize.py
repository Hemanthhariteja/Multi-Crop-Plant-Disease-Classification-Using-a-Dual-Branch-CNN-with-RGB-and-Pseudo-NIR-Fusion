import numpy as np
import matplotlib.pyplot as plt


def plot_rgb_nir(rgb, nir, ndvi=None):
    fig, axes = plt.subplots(1, 3 if ndvi is not None else 2, figsize=(10, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    axes[1].imshow(nir.squeeze(), cmap="gray")
    axes[1].set_title("NIR")
    axes[1].axis("off")

    if ndvi is not None:
        axes[2].imshow(ndvi.squeeze(), cmap="viridis")
        axes[2].set_title("NDVI")
        axes[2].axis("off")

    fig.tight_layout()
    return fig
