#!/usr/bin/env python3
import json

from cast import *

WEIGHTS = [1e1, 5e1, 1e2, 2e2]

STYLE_CONTENT_PAIRS = [
    ("lap_cartoon2", "girl2"),
    ("lap_girl3", "boy"),
    ("smallworld", "kid5")
]

exp = ExperimentRun('lapstyle-weights-1')

exp.dump_sources()

device = get_device()
print('Load VGG')
net = Vgg16().to(device)

fails = []

for style_name, content_name in STYLE_CONTENT_PAIRS:
    res_dest = exp.make_item_dir(f'{content_name}__{style_name}')
    content = exp.load_content(content_name, 512, res_dest)
    style = exp.load_style(style_name, 512, res_dest)

    for sw in WEIGHTS:
        res = res_dest / f'e={sw:.0e}.jpg'

        without_edge = perform_transfer(
            net, content, style,
            1e8,
            iterations=1000,
            device=device
        )

        save_image(without_edge, res_dest / 'noedge.jpg')

        try:
            result = perform_transfer(
                net, content, style,
                1e8,
                'atlap', dict(detector="lapstyle", normalize=False, reduction="mse"), sw,
                device=device
            )
            save_image(result, res)
        except Exception as e:
            print(e)
            fails.append(str(res))

json.dump(fails, (exp.results_dir / 'fails.json').open('w'), indent=2)
