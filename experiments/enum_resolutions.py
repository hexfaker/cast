#!/usr/bin/env python3
import os
import shutil
from datetime import datetime
import torch
from PIL import Image

from cast import *


assert os.path.exists('./runs'), "Check working dir"

RESOLUTIONS = [256, 512, 768, 1024]

CONTENT_IMAGES = [
   # 'bottles',
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

RESULT_PATH = f'runs/resolutions-{datetime.now().isoformat()}'

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
        
        for style_size in RESOLUTIONS:
            for content_size in RESOLUTIONS:
                
                content = load_image(f'images/content/{content_name}.jpg', content_size)
                style = load_image(f'images/styles/{style_name}.jpg', style_size)
                
                res = f'{image_dest}/c{content_size}__s{style_size}.jpg'

                try:
                    result = perform_transfer(
                        net, content, style,
                        1e5,
                        'sobel', {'normalize':False}, 1e-1,
                        device=device
                    )
                    save_image(result, res)
                except Exception as e:
                    print(e)
                    FAIL_IMAGE.save(res)



