from typing import Dict, Any

import torch
import tqdm

from . import loss

_EDGE_LOSSES = {
    'sobel': loss.SobelEdgeLoss,
    'bsobel': loss.BlurredSobelEdgeLoss
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

        if loss_value == float('inf'):
            raise Exception('inf occured in optimization')

        self.progress.update()
        self.progress.set_description(f'{loss_value:.5f}')

        return self.loss

    def run(self, steps_max):
        self.steps_run = 0
        self.progress = self.progress_factory(total=steps_max)

        while self.steps_run < steps_max:
            self.optim.step(self)

        self.progress.close()
        self.progress = None


def perform_transfer(
    net: torch.nn.Module, content_image: torch.Tensor, style_image: torch.Tensor,
    style_weight: float, edge_loss: str = None, edge_loss_params: Dict[str, Any] = None,
    edge_weight: float = 0, iterations=500, init='content', device=None,
    progressbar_factory=tqdm.tqdm
):
    content_image = content_image.to(device)
    style_image = style_image.to(device)
    if edge_loss_params is None:
        edge_loss_params = {}

    losses = [loss.ContentLoss(net, content_image).to(device),
              loss.StyleLoss(net, style_image).to(device)]
    loss_factors = [1., style_weight]

    if edge_loss is not None:
        losses.append(_EDGE_LOSSES[edge_loss](content_image, device=device, **edge_loss_params))
        loss_factors.append(edge_weight)

    total_loss = loss.LinearCombinationLoss(net, losses, loss_factors).to(device)

    runner = BfgsOptimizer(content_image, total_loss, progressbar_factory)

    runner.run(iterations)

    return runner.out.detach()
