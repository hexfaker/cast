import os
import argparse
from glob import glob

import torch

from .transfer import perform_transfer
from .vgg import Vgg16
from .utils import load_image, save_image


def file_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', required=True)
    parser.add_argument('--content', required=True)
    parser.add_argument('--edge-loss', choices=['sobel', 'bsobel'])
    parser.add_argument('--edge-loss-weight', '--elw', type=float)
    parser.add_argument('--style-loss-weight', '--slw', default='1e4', type=float)
    parser.add_argument('--content-loss-weight', '--clw', default='1e4', type=float)
    parser.add_argument('--results', required=True)
    parser.add_argument('--iterations', default=500, type=int)
    parser.add_argument('--style-size', default=256, type=int)
    parser.add_argument('--content-size', default=256, type=int)

    return parser.parse_args()


def perform(net, content_path, style_path, output_dir, style_height, content_height,
    style_factor, edge_loss, edge_loss_factor, iterations, device):
    content = load_image(content_path, content_height)
    style = load_image(style_path, style_height)

    content_name = file_name(content_path)
    style_name = file_name(style_path)
    res_filename = os.path.join(
        output_dir,
        f'{content_name}_x_{style_name}_{edge_loss}_{edge_loss_factor}.jpg'
    )

    res = perform_transfer(net, content, style, style_factor, edge_loss,
                           edge_loss_factor=edge_loss_factor, device=device,
                           iterations=iterations)

    save_image(res, res_filename)


def main():
    args = parse_args()

    if os.path.isdir(args.style):
        styles = glob(os.path.join(args.style, '*.jpg'))
    else:
        styles = [args.style]
    if os.path.isdir(args.content):
        contents = glob(os.path.join(args.content, '*.jpg'))
    else:
        contents = [args.content]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'Running on {device}')
    print('Load VGG')
    net = Vgg16().to(device)

    for style_path in styles:
        for content_path in contents:
            perform(
                net, content_path, style_path,
                args.results,
                args.style_size, args.content_size,
                args.style_loss_weight, args.edge_loss, args.edge_loss_weight,
                args.iterations,
                device
            )

main()
