import torch


def compute_ndvi(red, nir, eps=1e-6):
    return (nir - red) / (nir + red + eps)


def add_ndvi_band(image):
    red = image[:, 0:1, :, :]
    nir = image[:, 3:4, :, :]
    ndvi = compute_ndvi(red, nir)
    return torch.cat([image, ndvi], dim=1)
