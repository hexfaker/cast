#!/usr/bin/env python3
import os
import shutil
from datetime import datetime
import torch
from PIL import Image

from cast import *

assert os.path.exists('./runs'), "Check working dir"

RESOLUTIONS = [
    256,
    512,
    1024
]

STYLE_WEIGHT = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

CONTENT_IMAGES = [
    'bottles',
    'portrait6',
    'nature',
]

STYLE_IMAGES = [
    'abstract',
    'abstract1',
    'aqua0',
    'aqua1',
    'sketch',
]

RESULT_PATH = f'runs/sweights-{datetime.now().isoformat()}'

os.makedirs(RESULT_PATH)

def dump_sources():
    """Copies everything experiment depends on to make result reproducible"""

    dirs = ['cast', 'images', 'experiments']

    dest_path = f'{RESULT_PATH}/root'
    os.makedirs(dest_path)

    for d in dirs:
        shutil.copytree(d, f'{dest_path}/{d}')


FAIL_IMAGE = Image.new('RGB', (100, 100), (255, 0, 0))

print(f'Run dir: {RESULT_PATH}')
dump_sources()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f'Running on {device}')
print('Load VGG')
net = Vgg16().to(device)

for content_name in CONTENT_IMAGES:
    for style_name in STYLE_IMAGES:
        image_dest = f'{RESULT_PATH}/{content_name}__{style_name}'
        os.makedirs(image_dest)

        for size in RESOLUTIONS:
            for w in STYLE_WEIGHT:
                content = load_image(f'images/content/{content_name}.jpg', size)
                style = load_image(f'images/styles/{style_name}.jpg', size)

                res = f'{image_dest}/{w:.0e}__s{size}.jpg'

                print(res)

                result = perform_transfer(
                    net, content, style,
                    w,
                    init='cpn',
                    device=device
                )
                save_image(result, res)
