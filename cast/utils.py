import json
import shutil
from typing import Union
from pathlib import Path
from datetime import datetime

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

    image = image.byte().numpy()

    image = image.transpose(1, 2, 0)

    return image


def load_image(path: str, longest_side_max: int = None):
    """
    Load image as torch.Tensor. Optionally resize so that heigh is equal to specified

    :param path:
    :param longest_side_max: Maximum image dimension 
    :return:  image tensor of size (1x3xHxW)
    """

    image: Image.Image = Image.open(path).convert('RGB')

    if longest_side_max is not None:
        max_side = max(image.height, image.width)

        scale = longest_side_max / max_side
        new_height = int(image.height * scale)
        new_width = int(image.width * scale)
        image = image.resize((new_width, new_height), resample=Image.LANCZOS)

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


class ExperimentRun:
    def __init__(self, name):
        runs_dir = Path('runs')
        assert runs_dir.is_dir(), 'Check working dir'

        timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        self.results_dir = runs_dir / f'{name}__{timestamp}'

        self.results_dir.mkdir()

        print(f'Run dir: {self.results_dir}')

    def load_style(self, name, size, item_dir=None):
        path = f'images/styles/{name}.jpg'
        if item_dir:
            shutil.copy(path, item_dir / 'style.jpg')
        return load_image(path, size)

    def load_content(self, name, size, item_dir=None):
        path = f'images/content/{name}.jpg'
        if item_dir:
            shutil.copy(path, item_dir / 'content.jpg')
        return load_image(path, size)

    def dump_sources(self):
        """Copies everything experiment depends on to make result reproducible"""

        dirs = ['cast', 'images', 'experiments']

        dest_path = self.results_dir / 'root'
        dest_path.mkdir()

        for d in dirs:
            shutil.copytree(
                d, f'{dest_path}/{d}',
                ignore=lambda _, names: ['__pycache__'] if '__pycache__' in names else []
            )

    def make_item_dir(self, name):
        res = self.results_dir / name
        res.mkdir()

        return res

    def write_json(self, fname, obj):
        json.dump(obj, (self.results_dir / fname).open('w'), indent=2)


def get_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Running on {device}')
    return device
