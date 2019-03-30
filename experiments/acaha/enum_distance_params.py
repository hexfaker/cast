#!/usr/bin/env python3
from cast import *

SOBEL_WEIGHTS = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e5, 1e7, 1e9]
ALPHAS = [0.5, 1, 2, 3]

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

exp = ExperimentRun('acaha-distance-params')

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

        for a in ALPHAS:
            for edge_weight in SOBEL_WEIGHTS:
                res = res_dest / f'a={a}e={edge_weight:.0e}.jpg'

                try:
                    result = perform_transfer(
                        net, content, style,
                        1e8,
                        'acaha', dict(alpha=a, downscale_factor=2.5), edge_weight,
                        init='cpn',
                        device=device
                    )
                    save_image(result, res)
                except Exception as e:
                    print(e)
                    import traceback

                    traceback.print_exc()
                    fails.append(str(res))

exp.write_json('fails.json', fails)
