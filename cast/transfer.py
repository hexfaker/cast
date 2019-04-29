from typing import Dict, Any

import torch
from torch import nn
import tqdm

from cast.loss import StyleTransferLoss
from . import loss

_EDGE_LOSSES = {
    'sobel': loss.SobelEdgeLoss,
    'tsobel': loss.ThresholdedSobelEdgeLoss,
    'asoha': loss.AsymmetricSobelHausdorffLoss,
    'qsobel': loss.l2.QuantileAsymmetricSobelL2Loss,
    'acaha': loss.hausdorff.AsymmetricCannyHausdorffLoss,
    'lap': loss.LapStyleLoss,
    'atlap': loss.ParametricLaplaceEdgeLoss
}


class BfgsOptimizer:
    def __init__(self, init, loss_function, progress_factory=tqdm.tqdm, **optim_params):
        self.progress_factory = progress_factory

        self.out = init.clone()
        self.out.requires_grad = True

        self.loss_function = loss_function
        self.optim = torch.optim.LBFGS([self.out], **optim_params)

        self.steps_run = None
        self.loss = None

        self.progress = None

        self.out_hist = []
        self.loss_hist = []

    def release_gpu_resources(self):
        self.out = None
        self.loss_function = None
        self.optim = None
        self.loss = None

    def __call__(self):
        self.steps_run += 1

        self.optim.zero_grad()

        self.loss = self.loss_function(self.out)

        self.loss.backward()

        loss_value = self.loss.item()

        if torch.isfinite(self.loss).sum().item() == 0:
            raise Exception(f'{self.loss} occured in optimization')

        self.progress.update()
        self.progress.set_description(f'{loss_value:g}')

        return self.loss

    def run(self, steps_max):
        self.steps_run = 0
        self.progress = self.progress_factory(total=steps_max)

        while self.steps_run < steps_max:
            self.optim.step(self)

        self.progress.close()
        self.progress = None


class StyleTransferLossWrapper(nn.Module):
    def __init__(self, net: nn.Module, loss: StyleTransferLoss):
        super().__init__()
        self.loss = loss
        self.net = net

    def set_targets(self, content, style):
        self.loss.set_target(self.net, content, style)

    def forward(self, image):
        return self.loss(image, self.net(image))


def perform_transfer(
    net: torch.nn.Module, content_image: torch.Tensor, style_image: torch.Tensor,
    style_weight: float, edge_loss: str = None, edge_loss_params: Dict[str, Any] = None,
    edge_weight: float = 0, iterations=500, optim_params=None, init='content', device=None,
    progressbar_factory=tqdm.tqdm
):
    content_image = content_image.to(device)
    style_image = style_image.to(device)
    if edge_loss_params is None:
        edge_loss_params = {}

    optim_params = optim_params or {}

    losses = [loss.ContentLoss(), loss.StyleLoss()]
    loss_factors = [1., style_weight]

    if edge_loss is not None:
        losses.append(_EDGE_LOSSES[edge_loss](**edge_loss_params))
        loss_factors.append(edge_weight)

    total_loss = StyleTransferLossWrapper(
        net,
        loss.LinearCombinationLoss(losses, loss_factors)
    ).to(device)

    total_loss.set_targets(content_image, style_image)

    initial_image = content_image

    if init == 'cpn':
        initial_image += torch.randn_like(content_image) * 1e-3

    runner = BfgsOptimizer(initial_image, total_loss,
                           progressbar_factory,
                           **optim_params)

    runner.run(iterations)

    return runner.out.detach()
