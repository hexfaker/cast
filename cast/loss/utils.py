import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


def get_quantile(values: torch.Tensor, q: float):
    values = values.cpu().numpy().flatten()
    quantile = np.quantile(values, q)
    return quantile.item()


def threshold_by_quantile(values: torch.Tensor, q: float):
    quantile = get_quantile(values, q)

    values[values < quantile] = 0

    return values


class Normalizer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.denom = 1

    def fit_transform(self, t: torch.Tensor):
        assert not t.requires_grad, 'Target should be detached'

        if self.enabled:
            self.denom = t.max()
            t /= self.denom

        return t

    def transform(self, t: torch.Tensor):
        if self.enabled:
            return t / self.denom
        return t
