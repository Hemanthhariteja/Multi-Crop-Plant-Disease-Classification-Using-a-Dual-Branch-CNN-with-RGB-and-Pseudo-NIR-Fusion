import torch
from torch import nn


def get_classification_loss():
    return nn.CrossEntropyLoss()


def get_weighted_loss(class_weights):
    weights = torch.tensor(class_weights, dtype=torch.float)
    return nn.CrossEntropyLoss(weight=weights)
