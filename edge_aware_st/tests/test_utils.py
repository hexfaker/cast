import os
from tempfile import TemporaryDirectory

import numpy as np

from ..utils import ndimage2tensor, tensor2ndimage, save_image, load_image


def test_image_conversion():
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    tensor_image = ndimage2tensor(image)

    assert np.all(np.abs(image - tensor2ndimage(tensor_image)) <= 1)


def test_save_load():
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    tensor_image = ndimage2tensor(image)

    with TemporaryDirectory() as dir_path:
        fp = os.path.join(dir_path, 'iamge.png')
        save_image(tensor_image, fp)
        loaded_image = load_image(fp)

    assert np.allclose(loaded_image, tensor_image, atol=1 / 255)
