import torch
import torch.nn.functional as F
import torch.nn as nn

from cast.loss.common import StyleTransferLoss
from cast.loss.utils import Normalizer, threshold_by_quantile
from .flters import SobelFilter


def cdist(x, y):
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.norm(differences.float(), dim=-1, p=2)
    return distances


class AsymmetricHausdorfDistance(nn.Module):
    def __init__(self, input_hw, original_hw, target, alpha=4):
        """
        """
        super().__init__()

        # Prepare all possible (row, col) locations in the image
        self.target = target
        self.alpha = alpha
        original_hw = torch.tensor(original_hw)
        self.scale_factor = (original_hw / torch.tensor(input_hw)) \
            .view(1, 2).float().to(target.device)
        self.max_dist = torch.norm(original_hw.float(), p=2)

        input_locations = torch.stack(
            (torch.arange(input_hw[0])[:, None].repeat(1, input_hw[1]),
             torch.arange(input_hw[1])[None, :].repeat(input_hw[0], 1)),
            2).float().view(-1, 2).to(target.device) * self.scale_factor
        target *= self.scale_factor

        self.distances = cdist(input_locations, self.target)

    def forward(self, input: torch.Tensor):
        input = input.view(-1, 1)

        eps = 1e-6
        alpha = self.alpha

        denom = (self.distances + eps) / (input ** alpha + eps / self.max_dist)
        d_div_p = torch.min(denom, 0)

        d_div_p = torch.clamp(d_div_p[0], 0, self.max_dist)
        result = torch.mean(d_div_p, 0)
        return result


class AsymmetricSobelHausdorfLoss(StyleTransferLoss):
    def __init__(self, q, alpha, normalize=True, downscale_factor=2):
        super().__init__()
        self.alpha = alpha
        self.downscale_factor = downscale_factor
        self.normalizer = Normalizer(normalize)
        self.q = q
        self.sobel = SobelFilter(False)
        self.hausdorf: AsymmetricHausdorfDistance = None

    def _get_edges(self, image):
        contours = self.sobel(image)[0]
        contours = F.interpolate(
            contours,
            scale_factor=1 / self.downscale_factor,
            mode='bilinear',
            align_corners=True
        )
        return self.normalizer.transform(contours.squeeze(1))

    def set_target(self, net: nn.Module, content: torch.Tensor, style: torch.Tensor):
        original_hw = content.shape[-2:]
        with torch.no_grad():
            target = self._get_edges(content)
        target = self.normalizer.fit_transform(target)
        target = threshold_by_quantile(target, self.q)

        h, w = target.shape[-2:]

        coords = torch.stack((torch.arange(h)[:, None].repeat(1, w),
                              torch.arange(w)[None, :].repeat(h, 1)),
                             2).float().to(target.device)

        target = coords[target[0] > 0]
        self.hausdorf = AsymmetricHausdorfDistance((h, w), original_hw, target, self.alpha)

    def compute(self, image: torch.Tensor, _):
        contour = self._get_edges(image)

        return self.hausdorf(contour)
