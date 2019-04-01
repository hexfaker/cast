from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from cast.loss import StyleTransferLoss
from cast.loss.flters import LaplaceFilter


class LapStyleLoss(StyleTransferLoss):
    """
    Reimplementation of http://arxiv.org/abs/1707.01253
    """

    def __init__(self, pooling_kernel_size=2):
        super().__init__()
        self.edge_detector = nn.Sequential(
            nn.AvgPool2d(pooling_kernel_size),
            LaplaceFilter()
        )
        self.target = None

    def set_target(self, net: nn.Module, content: torch.Tensor, style: torch.Tensor):
        with torch.no_grad():
            self.target = self.edge_detector(content)

    def compute(self, image: torch.Tensor, features: List[torch.Tensor]):
        edges = self.edge_detector(image)
        return F.mse_loss(edges, self.target)
