import math

import torch
import torch.nn.functional as F
from torch import nn


def gauss(x, sigma):
    ss = 2 * sigma ** 2
    return 1 / (math.pi * ss) * torch.exp(-x / ss)


class ToGray(nn.Module):
    def __init__(self, mode='lum'):
        super().__init__()

        if mode == "lum":
            self.weights = (.2126, .7152, .0722)
        else:
            self.weights = (0.3333, )

    def f(self, image: torch.Tensor):
        w = image.new(self.weights).view(1, -1, 1, 1)
        res = (w * image).sum(-3, keepdim=True)
        return res

    def forward(self, image):
        return self.f(image)


class SobelFilter(nn.Module):
    """
    Input: Image in pytorch format ([1 x ] x 3 x W x H)
    Output: 2 x W x H (0 - magnitudes, 1 - angles)
    """

    # noinspection PyArgumentList
    @staticmethod
    def make_kernel(in_ch):
        sobel = torch.FloatTensor(
            [[1., 0., -1.],
             [2., 0., -2.],
             [1., 0., -1.]]
        )

        sobel_x_rgb = torch.stack([sobel] * in_ch, 0)
        sobel_y_rgb = torch.stack([sobel.t()] * in_ch, 0)

        return torch.stack([sobel_x_rgb, sobel_y_rgb])

    def __init__(self, angles=True, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.angles = angles
        self.k = nn.Parameter(self.make_kernel(self.in_channels), requires_grad=False)

    def forward(self, inp):
        sobel_xy = F.conv2d(inp, self.k)
        magnitude = torch.sqrt((sobel_xy ** 2).sum(dim=1, keepdim=True))

        if self.angles:
            angle = torch.atan2(sobel_xy[1], sobel_xy[0])
            return magnitude, angle
        else:
            return magnitude


class GaussFilter(nn.Module):
    @staticmethod
    def make_kernel(sigma):
        ks = math.ceil(6 * sigma)
        ks += 1 - ks % 2
        horizontal_idx = torch.arange(-(ks // 2), ks // 2 + 1).unsqueeze(0).float() ** 2
        vertical_idx = horizontal_idx.t()
        gk_one_plane = gauss(vertical_idx + horizontal_idx, sigma)

        zeros = torch.zeros(ks, ks)

        gk_r = torch.stack([gk_one_plane, zeros, zeros])
        gk_g = torch.stack([zeros, gk_one_plane, zeros])
        gk_b = torch.stack([zeros, zeros, gk_one_plane])

        return torch.stack([gk_r, gk_g, gk_b]), ks

    @staticmethod
    def _noop(x, *args, **kwargs):
        return x

    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.f = None
        if sigma > 0:
            self.k, self.size = self.make_kernel(sigma)
            self.k = nn.Parameter(self.k, requires_grad=False)
            self.f = F.conv2d
        else:
            self.size = 0
            self.k = None
            self.f = self._noop

    def forward(self, input):
        return self.f(input, self.k, padding=self.size // 2)


class LaplaceFilter(nn.Module):

    @staticmethod
    def _make_kernel():
        k = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float).expand(1, 3, 3, 3)
        return k

    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(self._make_kernel(), False)

    def forward(self, input):
        return F.conv2d(input, self.k)


class CannyEdgeDetector(nn.Module):
    """
    Input: SobelFilter output
    Output: Edge map
    """

    def __init__(self):
        super().__init__()
        self.lower_bounds = nn.Parameter(
            torch.linspace(-5 / 8 * math.pi, 3 / 8 * math.pi, 5).view(-1, 1, 1),
            requires_grad=False
        )
        self.upper_bounds = nn.Parameter(
            torch.linspace(-3 / 8 * math.pi, 5 / 8 * math.pi, 5).view(-1, 1, 1),
            requires_grad=False
        )

        self.x_shifts = [0, 1, 1, 1, 0]
        self.y_shifts = [1, 1, 0, -1, 1]

    def forward(self, inp: torch.Tensor):
        magnitude, angle = inp
        angle_segment_mask = (angle.unsqueeze(0) >= self.lower_bounds) & \
                             (angle.unsqueeze(0) < self.upper_bounds)  # type: torch.Tensor
        neighbours = F.pad(magnitude, [2] * 4)
        max_magnitudes = magnitude.clone()

        for i, (xs, ys) in enumerate(zip(self.x_shifts, self.y_shifts)):
            is_not_maximum_positive_direction = magnitude < neighbours[2 + ys: - 2 + ys,
                                                            2 + xs: -2 + xs]
            is_not_maximum_negative_drection = magnitude < neighbours[2 - ys: - 2 - ys,
                                                           2 - xs: -2 - xs]
            suppress = \
                (1 -
                 (angle_segment_mask[i] &
                  (is_not_maximum_negative_drection | is_not_maximum_positive_direction))
                 .float()
                 )
            max_magnitudes = suppress * max_magnitudes

        return max_magnitudes
