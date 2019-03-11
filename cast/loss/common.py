from abc import ABC, abstractmethod
from typing import List, Iterable

import torch
from torch import nn


class StyleTransferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def set_target(self, net: nn.Module, content: torch.Tensor, style: torch.Tensor):
        super()
        pass

    @abstractmethod
    def compute(self, image: torch.Tensor, features: List[torch.Tensor]):
        pass

    def forward(self, image, feautes):
        return self.compute(image, feautes)


class LinearCombinationLoss(StyleTransferLoss):
    def __init__(self, losses: Iterable[StyleTransferLoss], coefs: Iterable[float]):
        super().__init__()

        self.losses = nn.ModuleList(losses)  # type: Iterable[StyleTransferLoss]
        self.coefs = coefs

    def set_target(self, net: nn.Module, content: torch.Tensor, style: torch.Tensor):
        for l in self.losses:
            l.set_target(net, content, style)

    def compute(self, image: torch.Tensor, features: List[torch.Tensor]):
        sum = 0.
        for loss, coef in zip(self.losses, self.coefs):
            sum = sum + loss(image, features) * coef
        return sum
