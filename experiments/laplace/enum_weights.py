#!/usr/bin/env python3
import json

from cast import *

WEIGHTS = [1e0, 1e1, 1e2, 1e3, 1e5, 1e7, 1e9]

CONTENT_IMAGES = [
    'megan',
    'boy',
    'goat',
]

STYLE_IMAGES = [
    'la_muse',
    'flowers',
    'smallworld',
]

exp = ExperimentRun('lapstyle-weights')

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

        for sw in WEIGHTS:
            res = res_dest / f'e={sw:.0e}.jpg'

            without_edge = perform_transfer(
                net, content, style,
                1e8,
                device=device, init='cpn'
            )

            save_image(without_edge, res_dest / 'noedge.jpg')

            try:

                result = perform_transfer(
                    net, content, style,
                    1e8,
                    'lap', dict(), sw,
                    init='cpn',
                    device=device
                )
                save_image(result, res)
            except Exception as e:
                print(e)
                fails.append(str(res))

json.dump(fails, (exp.results_dir / 'fails.json').open('w'), indent=2)
