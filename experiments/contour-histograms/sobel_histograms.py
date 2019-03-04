from cast import ExperimentRun
from cast.loss import SobelFilter
from cast.utils import tensor2ndimage
import matplotlib.pyplot as plt

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

exp = ExperimentRun('sobel-histogram')
sobel = SobelFilter(angles=False)

for c in CONTENT:
    content = exp.load_content(c, size=512)

    contours = sobel(content)[0].numpy()

    plt.figure(121)
    plt.imshow(tensor2ndimage(content))
    plt.figure(122)
    plt.hist(contours.flatten(), bins=100)

    plt.savefig(exp.results_dir / f'{c}.png')
    plt.close()


