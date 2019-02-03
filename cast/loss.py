import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import math


def gram_matrix(y):
    """
    @from https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class ContentLoss(nn.Module):
    def __init__(self, net, content):
        super().__init__()
        self.net = net
        with torch.no_grad():
            self.target_feature = net(content)[2]

    def forward(self, img, features=None):
        if features is None:
            features = self.net(img)

        return F.mse_loss(features[2], self.target_feature)


class StyleLoss(nn.Module):
    def __init__(self, net, style):
        super().__init__()

        self.net = net
        with torch.no_grad():
            target_features = net(style)
        self.target_grams = [gram_matrix(f) for f in target_features]

    def forward(self, img, features=None):
        if features is None:
            features = self.net(img)

        input_grams = [gram_matrix(f) for f in features]
        loss = F.mse_loss(input_grams[0], self.target_grams[0])
        for i in range(1, len(input_grams)):
            loss = loss + F.mse_loss(input_grams[i], self.target_grams[i])
        return loss


def gauss(x, sigma):
    ss = 2 * sigma ** 2
    return 1 / (math.pi * ss) * torch.exp(-x / ss)


class SobelFilter(nn.Module):
    """
    Input: Image in pytorch format ([1 x ] x 3 x W x H)
    Output: 2 x W x H (0 - magnitudes, 1 - angles)
    """

    # noinspection PyArgumentList
    @staticmethod
    def make_kernel():
        sobel = torch.FloatTensor(
            [[1., 0., -1.],
             [2., 0., -2.],
             [1., 0., -1.]]
        )

        sobel_x_rgb = torch.stack([sobel] * 3, 0)
        sobel_y_rgb = torch.stack([sobel.t()] * 3, 0)

        return torch.stack([sobel_x_rgb, sobel_y_rgb])

    def __init__(self, angles=True):
        super().__init__()
        self.angles = angles
        self.k = Parameter(self.make_kernel(), requires_grad=False)

    def forward(self, inp):
        sobel_xy = F.conv2d(inp, self.k).squeeze()
        magnitude = torch.sqrt((sobel_xy ** 2).sum(dim=0))

        res = [magnitude]

        if self.angles:
            angle = torch.atan2(sobel_xy[1], sobel_xy[0])
            res.append(angle)

        return res


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

        return torch.stack([gk_r, gk_g, gk_b])

    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.k = Parameter(self.make_kernel(sigma), requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.k)


class CannyEdgeDetector(nn.Module):
    """
    Input: SobelFilter output
    Output: Edge map
    """

    def __init__(self):
        super().__init__()
        self.lower_bounds = Parameter(
            torch.linspace(-5 / 8 * math.pi, 3 / 8 * math.pi, 5).view(-1, 1, 1),
            requires_grad=False
        )
        self.upper_bounds = Parameter(
            torch.linspace(-3 / 8 * math.pi, 5 / 8 * math.pi, 5).view(-1, 1, 1),
            requires_grad=False
        )

        self.x_shifts = [0, 1, 1, 1, 0]
        self.y_shifts = [1, 1, 0, -1, 1]

    def forward(self, inp: torch.Tensor):
        magnitude, angle = inp
        angle_segment_mask = (angle.unsqueeze(0) >= self.lower_bounds) & \
                             (angle.unsqueeze(0) < self.upper_bounds)  # type: torch
        neighbours = F.pad(magnitude, tuple([2] * 4))
        max_magnitudes = magnitude.clone()

        for i, (xs, ys) in enumerate(zip(self.x_shifts, self.y_shifts)):
            is_not_maximum_positive_direction = magnitude < neighbours[2 + ys: - 2 + ys,
                                                            2 + xs: -2 + xs]
            is_not_maximum_negative_drection = magnitude < neighbours[2 - ys: - 2 - ys,
                                                           2 - xs: -2 - xs]
            suppress = (1 - (angle_segment_mask[i] &
                             (
                                 is_not_maximum_negative_drection | is_not_maximum_positive_direction)).float())
            max_magnitudes = suppress * max_magnitudes

        return max_magnitudes


class SobelEdgeLoss(nn.Module):
    def __init__(self, target_img, device, normalize=True):
        super().__init__()

        self.f = SobelFilter(angles=False)

        self.to(device)
        
        with torch.no_grad():
            self.target = self.f(target_img)[0]

            if normalize:
                self.norm_denominator = self.target.max()
                self.target = self.target / self.norm_denominator
            else:
                self.norm_denominator = 1.

    def forward(self, img, features=None):
        inp = self.f(img)[0] / self.norm_denominator
        return F.mse_loss(inp, self.target)


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


class CannyEdgeLoss(nn.Module):
    def __init__(self, target, device):
        super().__init__()

        self.f = nn.Sequential(GaussFilter(3), SobelFilter(), CannyEdgeDetector())
        self.to(device)

        with torch.no_grad():
            self.target = self.f(target)

    def forward(self, inp):
        return F.mse_loss(self.f(inp), self.target)


class LinearCombinationLoss(nn.Module):
    def __init__(self, net, losses, coefs):
        super().__init__()

        self.net = net
        self.losses = nn.ModuleList(losses)
        self.coefs = coefs

    def forward(self, img):
        features = self.net(img)
        sum = 0.
        for loss, coef in zip(self.losses, self.coefs):
            sum = sum + loss(img, features) * coef
        return sum
