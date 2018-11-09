from typing import Union

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F

_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406))
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225))


def ndimage2tensor(image: np.ndarray) -> torch.Tensor:
    image = F.to_tensor(image)
    image: torch.Tensor = F.normalize(image, _IMAGENET_MEAN, _IMAGENET_STD)
    image.unsqueeze_(0)
    return image


def tensor2ndimage(image: torch.Tensor) -> np.ndarray:
    image = image.clone().cpu().squeeze(0)
    for t, m, s in zip(image, _IMAGENET_MEAN, _IMAGENET_STD):
        t.mul_(s).add_(m)
    image.mul_(255)
    image.round_()
    image.clamp_(0, 255)

    image = image.byte().cpu().numpy()

    image = image.transpose(1, 2, 0)

    return image


def load_image(path: str, height: int = None):
    """
    Load image as torch.Tensor. Optionally resize so that heigh is equal to specified

    :param path:
    :param height:
    :return:  image tensor of size (1x3xHxW)
    """

    image: Image.Image = Image.open(path).convert('RGB')

    if height is not None:
        scale = height / image.height
        new_width = int(image.width * scale)
        image = image.resize((new_width, height), resample=Image.LANCZOS)

    return ndimage2tensor(np.array(image))


def save_image(image: Union[torch.tensor, np.ndarray], path: str):
    """
    Saves image
     
    :param image:
    :param path:
    :return:
    """
    if torch.is_tensor(image):
        image = tensor2ndimage(image)

    Image.fromarray(image).save(path)
