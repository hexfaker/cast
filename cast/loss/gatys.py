from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from cast.loss import StyleTransferLoss


def gram_matrix(y: torch.Tensor):
    """
    @from https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer
    """
    (b, ch, h, w) = y.shape
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class ContentLoss(StyleTransferLoss):

    def __init__(self):
        super().__init__()
        self.target_feature = None

    def set_target(self, net: nn.Module, content: torch.Tensor, style: torch.Tensor):
        with torch.no_grad():
            self.target_feature = net(content)[2]

    def compute(self, image: torch.Tensor, features: List[torch.Tensor]):
        return F.mse_loss(features[2], self.target_feature)


class StyleLoss(StyleTransferLoss):
    def __init__(self):
        super().__init__()
        self.target_grams = None

    def set_target(self, net: nn.Module, content: torch.Tensor, style: torch.Tensor):
        with torch.no_grad():
            target_features = net(style)
        self.target_grams = [gram_matrix(f) for f in target_features]

    def compute(self, image: torch.Tensor, features: List[torch.Tensor]):
        input_grams = [gram_matrix(f) for f in features]
        loss = ((input_grams[0] - self.target_grams[0]) ** 2).sum()
        for i in range(1, len(input_grams)):
            loss = loss + ((input_grams[i] - self.target_grams[i]) ** 2).sum()
        loss /= len(input_grams)
        return loss
