#!/usr/bin/env python3
from cast import *

THRESHOLDS = [.4, .5, .7, .9]
SOBEL_WEIGHTS = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]

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

exp = ExperimentRun('soha-weights-2')

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

        for edge_weight in SOBEL_WEIGHTS:
            for t in THRESHOLDS:
                res = res_dest / f'e={edge_weight:.0e}t={t:.1}.jpg'

                print(res)

                try:
                    result = perform_transfer(
                        net, content, style,
                        1e8,
                        'soha', dict(threshold=t), edge_weight,
                        init='cpn',
                        device=device
                    )
                    save_image(result, res)
                except Exception as e:
                    print(e)
                    fails.append(str(res))

exp.write_json('fails.json', fails)
