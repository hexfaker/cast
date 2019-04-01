from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from cast.loss.utils import Normalizer, threshold_by_quantile
from .common import StyleTransferLoss
from .flters import SobelFilter, GaussFilter, CannyEdgeDetector


class SobelEdgeLoss(nn.Module):
    def __init__(self, target_img, device, normalize=True, threshold=0):
        super().__init__()

        self.threshold = threshold
        self.f = SobelFilter(angles=False)

        self.to(device)

        with torch.no_grad():
            self.target = self.f(target_img)[0]

            if normalize:
                self.norm_denominator = self.target.max()
                self.target = self.target / self.norm_denominator
            else:
                self.norm_denominator = 1.

            self.target[self.target < self.threshold] = 0

    def forward(self, img, features=None):
        inp = self.f(img)[0] / self.norm_denominator
        inp[inp < self.threshold] = 0
        res = F.mse_loss(inp, self.target)
        return res


class BlurredSobelEdgeLoss(nn.Module):
    def __init__(self, target, device, sigma=3, threshold=.25):
        super().__init__()

        self.threshold = threshold
        self.f = SobelFilter(angles=False)
        self.blur = GaussFilter(sigma)
        self.to(device)

        with torch.no_grad():
            self.target = self.apply(target)

            self.norm_denominator = self.target.max()
            self.target = self.target / self.norm_denominator

    def apply(self, img):
        img = self.blur(img)
        img = self.f(img)[0]

        mask = img < self.threshold
        img = img.clone()
        img[mask] = 0.

        return img

    def forward(self, img, features=None):
        inp = self.apply(img) / self.norm_denominator
        return F.mse_loss(inp, self.target)


class ThresholdedSobelEdgeLoss(nn.Module):
    def __init__(self, target_img, device, normalize=True, threshold=0):
        super().__init__()

        self.threshold = threshold
        self.f = SobelFilter(angles=False)

        self.to(device)

        with torch.no_grad():
            self.target = self.f(target_img)[0]

            if normalize:
                self.norm_denominator = self.target.max()
                self.target = self.target / self.norm_denominator
            else:
                self.norm_denominator = 1.

            self.target[self.target < self.threshold] = 0

    def forward(self, img, features=None):
        inp = self.f(img)[0] / self.norm_denominator

        res = (torch.max(
            input=(self.target - inp),
            other=torch.zeros_like(self.target)
        ) ** 2).mean()
        return res


class QuantileAsymmetricSobelL2Loss(StyleTransferLoss):
    def __init__(self, q, sigma=0., normalize=True):
        super().__init__()
        self.normalizer = Normalizer(normalize)
        self.q = q
        self.sobel = SobelFilter(False)
        self.target = None
        self.blur = GaussFilter(sigma)

    def _get_edges(self, image):
        return self.normalizer.transform(self.sobel(image)[0])

    def set_target(self, net: nn.Module, content: torch.Tensor, style: torch.Tensor):
        with torch.no_grad():
            blur = self.blur(content)
            target = self._get_edges(blur)
        target = self.normalizer.fit_transform(target)
        self.target = threshold_by_quantile(target, self.q)

    @staticmethod
    def _asymmetric_l2(input, target):
        diff = target - input
        diff = torch.max(input=diff, other=torch.zeros_like(diff)) ** 2
        res = diff.mean()
        return res

    def compute(self, image: torch.Tensor, _: List[torch.Tensor]):
        edges = self._get_edges(image)
        res = self._asymmetric_l2(edges, self.target)
        return res


class CannyEdgeLoss(nn.Module):
    def __init__(self, target, device):
        super().__init__()

        self.f = nn.Sequential(GaussFilter(3), SobelFilter(), CannyEdgeDetector())
        self.to(device)

        with torch.no_grad():
            self.target = self.f(target)

    def forward(self, inp):
        return F.mse_loss(self.f(inp), self.target)
