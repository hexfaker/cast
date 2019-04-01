"""Run optimization for all images with thresholded sobel loss (weight=1e9, threshold=0.5) and
without. Sytle weight 1e8"""
import json

STYLE_WEIGHT = 1e8
EDGE_WEIGHT = [1e8, 5e8, 1e9]
SIGMA = [0, 1, 3, 5, 7, 10]
EDGE_THRESHOLD = [.7, .8, 0.9]

from cast import ExperimentRun, perform_transfer, Vgg16, get_device, save_image

CONTENT = [
    'details2',
    'details1',
    'portrait2',
    'portrait8',
]

STYLES = [
    'aqua_puantilism_2',
    'candy',
    'hadrd_geometry',
    'hard_geometry_2',
]

exp = ExperimentRun('qbsobel_sigma')

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

        for ew in EDGE_WEIGHT:
            for sig in SIGMA:
                for et in EDGE_THRESHOLD:
                    name = f"w={ew:g}s={sig}t={et}.jpg"
                    try:
                        with_edge = perform_transfer(
                            vgg, content, style,
                            STYLE_WEIGHT,
                            'qsobel', dict(sigma=sig, q=et), ew,
                            device=device, init='cpn'
                        )
                        res_path = item_dir / name
                        save_image(with_edge, res_path)
                    except Exception as e:
                        print(e)
                        fails.append(item_name + "/" + name)

json.dump(fails, (exp.results_dir / 'fails.json').open('w'), indent=2)
