from .. import loss
import torch


def test_sobel():
    m = loss.SobelFilter(False)

    res, = m(torch.zeros((1, 3, 100, 100)))

    assert res.shape[:2] == (1, 1)


def test_to_device():
    m = loss.GaussFilter(3)
    m = m.cuda()

    assert m.k.device.type == 'cuda'

    m = loss.SobelEdgeLoss(torch.zeros((1, 3, 100, 100)).cuda(), device=torch.device('cuda'))
    m = m.cuda()

    assert m.f.k.device.type == 'cuda'

    m = loss.SobelFilter(False)
    m.to(torch.device('cuda'))
    assert m.k.device.type == 'cuda'
