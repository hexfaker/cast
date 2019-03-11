#!/usr/bin/env python3
from cast import *

THRESHOLDS = [.1, .2, .5, .7, .9]
SOBEL_WEIGHTS = [1e5, 1e7, 1e9, 1e11]

CONTENT_IMAGES = [
    'bottles',
    'portrait6',
    'nature',
]

STYLE_IMAGES = [
    'abstract',
    'aqua1',
    'sketch',
]

exp = ExperimentRun('qsobel-weights-2')

exp.dump_sources()

device = get_device()
print('Load VGG')
net = Vgg16().to(device)

for content_name in CONTENT_IMAGES:
    for style_name in STYLE_IMAGES:
        res_dest = exp.make_item_dir(f'{content_name}__{style_name}')
        content = exp.load_content(content_name, 512, res_dest)
        style = exp.load_style(style_name, 512, res_dest)

        for sw in SOBEL_WEIGHTS:
            for t in THRESHOLDS:
                res = res_dest / f'e={sw:.0e}t={t:.1}.jpg'

                print(res)

                result = perform_transfer(
                    net, content, style,
                    1e8,
                    'qsobel', dict(q=t), sw,
                    init='cpn',
                    device=device
                )
                save_image(result, res)
