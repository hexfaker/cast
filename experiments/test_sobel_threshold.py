#!/usr/bin/env python3
import os
import torch

from cast import *

STYLE_WEIGHT = [1e8, 1e10]
THRESHOLDS = [.2, .5, .7, 1, 1.2, 1.5, 1.7]

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

exp = ExperimentRun('sobel-thresholds')

exp.dump_sources()

device = get_device()
print('Load VGG')
net = Vgg16().to(device)

for content_name in CONTENT_IMAGES:
    for style_name in STYLE_IMAGES:
        res_dest = exp.make_item_dir(f'{content_name}__{style_name}')
        content = exp.load_content(content_name, 512, res_dest)
        style = exp.load_style(style_name, 512, res_dest)

        for w in STYLE_WEIGHT:
            for t in THRESHOLDS:
                res = res_dest / f'{w:.0e}.jpg'

                print(res)

                result = perform_transfer(
                    net, content, style,
                    w,
                    'tsobel', 
                    init='cpn',
                    device=device
                )
                save_image(result, res)
