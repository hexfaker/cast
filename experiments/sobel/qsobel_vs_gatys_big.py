"""Run optimization for all images with thresholded sobel loss (weight=1e9, threshold=0.5) and
without. Sytle weight 1e8"""
import json

STYLE_WEIGHT = 1e8
EDGE_WEIGHT = [1e8, 5e8, 1e9]
EDGE_THRESHOLD = 0.9

from cast import ExperimentRun, perform_transfer, Vgg16, get_device, save_image

CONTENT = [
    'photo1',
    'details2',
    'details1',
    'details3',
    'details4',
    'portrait1',
    'portrait2',
    'portrait4',
    'portrait3',
    'portrait5',
    'portrait6',
    'portrait7',
    'portrait9',
    'portrait10',
    'portrait8',
    'bottles',
]

STYLES = [
    'aqua_puantilism_2',
    'geometry13',
    'candy',
    'aqua0',
    'hadrd_geometry',
    'abstract',
    'oil_strikes',
    'hard_geometry',
    'la_muse',
    'hard_geometry_2',
    'aqua1',
    'modern_abstract',
    'hard_geometry_3',
    'mosaics5',
    'lowdetail_aqua',
    'geometry_3d'
]

exp = ExperimentRun('qsobel_vs_gatys_big')

exp.dump_sources()

device = get_device()

vgg = Vgg16().to(device)

fails = []

for s in STYLES:
    for c in CONTENT:
        item_name = f'{s}__{c}'

        if (exp.results_dir / item_name).exists():
            continue

        item_dir = exp.make_item_dir(item_name)
        content = exp.load_content(c, 512, item_dir)
        style = exp.load_style(s, 512, item_dir)

        without_edge = perform_transfer(
            vgg, content, style,
            1e8,
            device=device, init='cpn'
        )

        save_image(without_edge, item_dir / 'noedge.jpg')

        for ew in EDGE_WEIGHT:
            try:
                with_edge = perform_transfer(
                    vgg, content, style,
                    STYLE_WEIGHT,
                    'qsobel', dict(q=EDGE_THRESHOLD), ew,
                    device=device, init='cpn'
                )
                res_path = item_dir / f'edge_{ew:g}.jpg'
                save_image(with_edge, res_path)
            except Exception as e:
                print(e)
                fails.append(str(item_name))

json.dump(fails, (exp.results_dir / 'fails.json').open('w'), indent=2)
