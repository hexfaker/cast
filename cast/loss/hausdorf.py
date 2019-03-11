import torch
import torch.nn.functional as F
import torch.nn as nn
import math

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
    distances = torch.sum(differences ** 2, -1).sqrt()
    return distances


def _assert_no_grad(variables):
    for var in variables:
        assert not var.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"


class WeightedHausdorffDistance(nn.Module):
    """From https://github.com/HaipengXiong/weighted-hausdorff-loss/blob/master/object-locator/losses.py"""

    def __init__(self,
        resized_height, resized_width,
        return_2_terms=False,
        alpha=4,
        device=torch.device('cpu')
    ):
        """
        :param resized_height: Number of rows in the image.
        :param resized_width: Number of columns in the image.
        :param return_2_terms: Whether to return the 2 terms
                               of the WHD instead of their sum.
                               Default: False.
        :param device: Device where all Tensors will reside.
        """
        super().__init__()

        # Prepare all possible (row, col) locations in the image
        self.alpha = alpha
        self.height, self.width = resized_height, resized_width
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)
        self.max_dist = math.sqrt(resized_height ** 2 + resized_width ** 2)
        self.n_pixels = resized_height * resized_width

        # Convert to appropiate type
        self.all_img_locations = torch.stack(
            (torch.arange(resized_height)[:, None].repeat(1, resized_width),
             torch.arange(resized_width)[None, :].repeat(resized_height, 1)),
            2).float().view(-1, 2).to(device)

        self.return_2_terms = return_2_terms

    def forward(self, prob_map, gt, orig_sizes):
        """
        Compute the Weighted Hausdorff Distance function
         between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.
        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :param orig_sizes: Bx2 Tensor containing the size of the original images.
                           B is batch size. The size must be in (height, width) format.
        :param orig_widths: List of the original width for each image in the batch.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                 If self.return_2_terms=True, then return a tuple containing
                 the two terms of the Weighted Hausdorff Distance.
        """

        _assert_no_grad(gt)

        if len(prob_map.shape) == 4:
            prob_map.squeeze_(1)

        assert prob_map.dim() == 3, 'The probability map must be (B x H x W)'
        assert prob_map.size()[1:3] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s' \
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == len(gt)

        terms_1 = []
        terms_2 = []
        for b in range(batch_size):
            # One by one
            prob_map_b = prob_map[b, :, :]
            gt_b = gt[b]
            orig_size_b = orig_sizes[b]
            norm_factor = (self.resized_size.new_tensor(orig_size_b) /
                           self.resized_size).unsqueeze(0)

            # Pairwise distances between all possible locations and the GTed locations
            n_gt_pts = gt_b.size()[0]
            normalized_x = norm_factor.repeat(self.n_pixels, 1) * self.all_img_locations
            normalized_y = norm_factor.repeat(len(gt_b), 1) * gt_b
            d_matrix = cdist(normalized_x, normalized_y)

            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())
            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            eps = 1e-6
            alpha = self.alpha

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + eps)) * \
                     torch.sum(p * torch.min(d_matrix, 1)[0])
            d_div_p = torch.min((d_matrix + eps) /
                                (p_replicated ** alpha + eps / self.max_dist), 0)[0]
            d_div_p = torch.clamp(d_div_p, 0, self.max_dist)
            term_2 = torch.mean(d_div_p, 0)

            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        if self.return_2_terms:
            res = terms_1.mean(), terms_2.mean()
        else:
            res = terms_1.mean() + terms_2.mean()

        return res


class SobelHausdorfLoss(nn.Module):
    def __init__(self, target_img, device, normalize=True, alpha=4, threshold=0,
        downscale_factor=2):
        super().__init__()
        self.alpha = alpha
        self.scale_factor = 1 / downscale_factor
        self.threshold = threshold
        self.sobel = SobelFilter(angles=False)
        self.denom = 1.

        self.to(device)

        with torch.no_grad():
            self.target = self._get_contour_map(target_img)

        if normalize:
            self.denom = self.target.max()
            self.target /= self.denom

        h, w = self.target.shape[-2:]

        coords = torch.stack((torch.arange(h)[:, None].repeat(1, w),
                              torch.arange(w)[None, :].repeat(h, 1)),
                             2).float()

        self.target = coords[self.target[0] > self.threshold].to(device)

        self.dist = WeightedHausdorffDistance(h, w, device=device)

    def _get_contour_map(self, image, features=None):
        contours = self.sobel(image)[0]
        contours = F.interpolate(contours, scale_factor=self.scale_factor, mode='bilinear',
                                 align_corners=True)
        contours.squeeze_(1)
        contours /= self.denom
        return contours

    def forward(self, image, features=None):
        contours = self._get_contour_map(image) / self.denom
        return self.dist(contours, [self.target], (image.shape[-2:],))


class AsymetricSobelHausdorfLoss(nn.Module):
    def __init__(self, target_img, device, alpha=4,
        normalize=True, threshold=0, downscale_factor=2):
        super().__init__()
        self.scale_factor = 1 / downscale_factor
        self.alpha = alpha
        self.threshold = threshold
        self.sobel = SobelFilter(angles=False)
        self.denom = 1.

        self.to(device)

        with torch.no_grad():
            self.target = self._get_contour_map(target_img)

        if normalize:
            self.denom = self.target.max()
            self.target /= self.denom

        h, w = self.target.shape[-2:]

        coords = torch.stack((torch.arange(h)[:, None].repeat(1, w),
                              torch.arange(w)[None, :].repeat(h, 1)),
                             2).float()

        self.target = coords[self.target[0] > self.threshold].to(device)

        self.dist = WeightedHausdorffDistance(h, w, alpha=self.alpha, return_2_terms=True,
                                              device=device)

    def _get_contour_map(self, image, features=None):
        contours = self.sobel(image)[0]
        contours = F.interpolate(contours, scale_factor=self.scale_factor, mode='bilinear',
                                 align_corners=True)
        contours.squeeze_(1)
        contours /= self.denom
        return contours

    def forward(self, image, features=None):
        contours = self._get_contour_map(image) / self.denom
        return self.dist(contours, [self.target], (image.shape[-2:],))[1]
