from .. import loss
import torch


def test_to_device():
    m = loss.GaussFilter(3)
    m = m.cuda()

    assert m.k.device.type == 'cuda'

    m = loss.SobelEdgeLoss(torch.zeros((1, 3, 100, 100)))
    m = m.cuda()

    assert m.f.k.device.type == 'cuda'
