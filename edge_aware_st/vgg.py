import torchvision
from torch import nn


class Vgg16(nn.Module):

    def __init__(self):
        super().__init__()

        layers = torchvision.models.vgg16(pretrained=True).features

        layer_groups = self.split_vgg(layers)[:4]

        self.relu1_2, self.relu2_2, self.relu3_3, self.relu4_3 = \
            [nn.Sequential(*group) for group in layer_groups]

        self.eval()

        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def split_vgg(layers: nn.Sequential):
        result = [[]]
        for layer in layers:
            if type(layer) is nn.MaxPool2d:
                result.append([layer])
            else:
                result[-1].append(layer)

        return result

    def forward(self, image):
        relu1_2 = self.relu1_2(image)
        relu2_2 = self.relu2_2(relu1_2)
        relu3_3 = self.relu3_3(relu2_2)
        relu4_3 = self.relu4_3(relu3_3)

        return relu1_2, relu2_2, relu3_3, relu4_3
