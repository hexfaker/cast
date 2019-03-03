#!/usr/bin/env python3
import json

from cast import *

THRESHOLDS = [.05, .2, .5, .7, 1., 1.5, 2., 2.5]
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

exp = ExperimentRun('sobel-weights-2-nonorm')

exp.dump_sources()

device = get_device()
print('Load VGG')
net = Vgg16().to(device)

fails = []
for content_name in CONTENT_IMAGES:
    for style_name in STYLE_IMAGES:
        res_dest = exp.make_item_dir(f'{content_name}__{style_name}')
        content = exp.load_content(content_name, 512, res_dest)
        style = exp.load_style(style_name, 512, res_dest)

        for sw in SOBEL_WEIGHTS:
            for t in THRESHOLDS:
                res = res_dest / f'e={sw:.0e}t={t:.1}.jpg'

                print(res)

                try:
                    result = perform_transfer(
                        net, content, style,
                        1e8,
                        'tsobel', dict(threshold=t, normalize=False), sw,
                        init='cpn',
                        device=device
                    )
                    save_image(result, res)

                except Exception as e:
                    print(e)
                    fails.append(res)

json.dump(fails, (exp.results_dir / 'fails.json').open('w'), indent=2)
