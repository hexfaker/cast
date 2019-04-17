from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from imageio import imread

TITILES = {
    'content': "Content",
    'style': "Style",
    'gatys': "Gatys et al",
    'lapstyle': "Lapstyle",
    'our': 'Ours'
}


def get_content_size(path):
    image = imread(path / "content.jpg")

    return image.shape[:2]


def get_grid_params(content_h, content_w):
    ar = content_h / content_w

    if ar > 1:
        shape = (2, 4)
        subplot_args = dict(
            style=dict(loc=(0, 0)),
            content=dict(loc=(1, 0)),
            gatys=dict(loc=(0, 1), rowspan=2),
            lapstyle=dict(loc=(0, 2), rowspan=2),
            our=dict(loc=(0, 3), rowspan=2)
        )
    else:
        shape = (2, 3)
        subplot_args = dict(
            style=dict(loc=(0, 0)),
            content=dict(loc=(1, 0)),
            gatys=dict(loc=(0, 1)),
            lapstyle=dict(loc=(1, 1)),
            our=dict(loc=(0, 2), rowspan=2)
        )

    return shape, subplot_args


def plot_images(path):
    plt.figure(dpi=200, figsize=(16, 9))

    images = get_image_paths(path)
    grid_shape, grid_args = get_grid_params(*get_content_size(path))

    for k, im in images.items():
        grid_imshow(im, TITILES[k], shape=grid_shape, **grid_args[k])

    plt.tight_layout(w_pad=0.3, h_pad=.8)

    plt.savefig(
        str(path / 'joint.pdf'),
        bbox_inches='tight'
    )


def grid_imshow(path, title='', **kwargs):
    plt.subplot2grid(**kwargs)
    plt.imshow(imread(path))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


def get_image_paths(result_dir: Path):
    our, lapstyle = list(result_dir.glob("edge*.jpg"))

    if "+" in str(lapstyle):
        our, lapstyle = lapstyle, our

    res = dict(
        content=result_dir / 'content.jpg',
        style=result_dir / 'style.jpg',
        gatys=result_dir / 'noedge.jpg',
        lapstyle=lapstyle,
        our=our
    )

    return res

parser = ArgumentParser()

parser.add_argument("path", type=Path)

args = parser.parse_args()

plot_images(args.path)
