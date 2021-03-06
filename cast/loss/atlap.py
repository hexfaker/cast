from typing import List

import torch
from torch import nn
from torch.nn.functional import mse_loss

from cast.loss import StyleTransferLoss
from cast.loss.flters import LaplaceFilter, SobelFilter, ToGray
from cast.loss.utils import threshold_by_quantile, Normalizer


def asymmetric_mse_loss(input, target):
    diff = target - input
    diff = torch.max(input=diff, other=torch.zeros_like(diff)) ** 2
    res = diff.mean()
    return res


_REDUCTION = {
    "mse": mse_loss,
    "amse": asymmetric_mse_loss
}


class _LapstyleEdgeDetector(nn.Module):
    def __init__(self, pooling_kernel=2):
        super().__init__()
        self.m = nn.Sequential(
            nn.AvgPool2d(pooling_kernel),
            LaplaceFilter()
        )

    def forward(self, image: torch.Tensor, is_stylization=False):
        return self.m(image)


class _LaplaceWithThreshold(nn.Module):
    def __init__(self, quantile=0.9):
        super().__init__()
        self.quantile = quantile
        self.laplace = LaplaceFilter()

    def forward(self, image: torch.Tensor, is_stylization=False):
        edges: torch.Tensor = self.laplace(image)

        if not is_stylization:
            edges = threshold_by_quantile(edges, self.quantile)

        return edges


class _SobelEdgeDetector(nn.Module):
    def __init__(self, quantile=0.9):
        super().__init__()
        self.quantile = quantile
        self.sobel = SobelFilter(angles=False)

    def forward(self, image: torch.Tensor, is_stylization=False):
        edges: torch.Tensor = self.sobel(image)[0]

        if not is_stylization:
            edges = threshold_by_quantile(edges, self.quantile)

        return edges


class _GSSobelEdgeDetector(nn.Module):
    def __init__(self, quantile=0.9, mode="lum"):
        super().__init__()
        self.quantile = quantile
        self.f = nn.Sequential(
            ToGray(mode),
            SobelFilter(angles=False, in_channels=1),
        )

    def forward(self, image: torch.Tensor, is_stylization=False):
        edges: torch.Tensor = self.f(image)[0]

        if not is_stylization:
            edges = threshold_by_quantile(edges, self.quantile)

        return edges


_EDGE_DETECTION = {
    "lapstyle": _LapstyleEdgeDetector,
    "laplace_qt": _LaplaceWithThreshold,
    "sobel_qt": _SobelEdgeDetector,
    "sobel_gs": _GSSobelEdgeDetector
}


class ParametricLaplaceEdgeLoss(StyleTransferLoss):
    def __init__(self, detector="laplace_qt", normalize=True, reduction="amse",
        norm_q=0, **detector_args):
        super().__init__()
        self.detector = _EDGE_DETECTION[detector](**detector_args)
        self.normalize = Normalizer(normalize, norm_q)
        self.reduction = _REDUCTION[reduction]

        self.target = None

    def compute(self, image: torch.Tensor, features: List[torch.Tensor]):
        edges = self.detector(image, is_stylization=True)
        edges = self.normalize.transform(edges)
        loss = self.reduction(edges, self.target)
        return loss

    def set_target(self, net: nn.Module, content: torch.Tensor, style: torch.Tensor):
        target = self.detector(content)
        self.target = self.normalize.fit_transform(target)
