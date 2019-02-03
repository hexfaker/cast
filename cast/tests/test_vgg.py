from ..vgg import Vgg16

def test_vgg16_load():
    vgg = Vgg16()

    lengths = [len(seq) for seq in (vgg.relu1_2, vgg.relu2_2, vgg.relu3_3, vgg.relu4_3)]

    assert lengths == [4, 5, 7, 7]
