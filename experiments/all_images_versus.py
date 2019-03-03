"""Run optimization for all images with thresholded sobel loss (weight=1e9, threshold=0.5) and
without. Sytle weight 1e8"""
import json

STYLE_WEIGHT = 1e8
EDGE_WEIGHT = 1e9
EDGE_THRESHOLD = 0.5

from cast import ExperimentRun, perform_transfer, Vgg16, get_device, save_image

CONTENT = [
    'bottles',
    'car1',
    'dark',
    'foots',
    'nat2',
    'nature',
    'photo1',
    'portrait1',
    'portrait2',
    'portrait3',
    'portrait4',
    'portrait5',
    'portrait6'
]

STYLES = [
    'abstract',
    'abstract1',
    'aqua0',
    'aqua1',
    'sketch'
]

exp = ExperimentRun('all-images-versus')

exp.dump_sources()

device = get_device()

vgg = Vgg16().to(device)

fails = []

for s in STYLES:
    for c in CONTENT:
        item_name = f'{s}__{c}'
        item_dir = exp.make_item_dir(item_name)
        content = exp.load_content(c, 512, item_dir)
        style = exp.load_style(s, 512, item_dir)

        without_edge = perform_transfer(
            vgg, content, style,
            1e8,
            device=device, init='cpn'
        )

        save_image(without_edge, item_dir / 'noedge.jpg')


        try:
            with_edge = perform_transfer(
                vgg, content, style,
                STYLE_WEIGHT,
                'tsobel', dict(threshold=EDGE_THRESHOLD), EDGE_WEIGHT,
                device=device, init='cpn'
            )
            save_image(with_edge, item_dir / 'edge.jpg')
        except Exception as e:
            print(e)
            fails.append(item_name)

json.dump(fails, (exp.results_dir / 'fails.json').open('w'), indent=2)





