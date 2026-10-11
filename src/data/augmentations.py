import torch


def random_flip(batch):
    if torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=[3])
    if torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=[2])
    return batch
