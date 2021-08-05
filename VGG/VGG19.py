from torch import nn


class VGG(nn.Module):
    def __init__(self, layer_list):
        super(VGG, self).__init__()
        self.features = self.make_layer(layer_list)
        self.classifier = nn.Linear(512, 10)

    def make_layer(self, layer_list):
        layers = []
        in_channel = 3
        for layer in layer_list:
            if layer == "m":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channel, layer, kernel_size=3, padding=1),
                    nn.BatchNorm2d(layer),
                    nn.ReLU(),
                ]
                in_channel = layer
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


VGG = {
    11: [64, "m", 128, "m", 256, 256, "m", 512, 512, "m", 512, 512, "m"],
    13: [64, 64, "m", 128, 128, "m", 256, 256, "m", 512, 512, "m", 512, 512, "m"],
    16: [64, 64, "m", 128, 128, "m", 256, 256, 256, "m", 512, 512, 512, "m", 512, 512, 512, "m"],
    19: [64, 64, "m", 128, 128, "m", 256, 256, 256, 256, "m", 512, 512, 512, 512, "m", 512, 512, 512, 512, "m"]
}


def VGG11():
    return VGG(11)


def VGG13():
    return VGG(13)


def VGG16():
    return VGG(16)


def VGG19():
    return VGG(19)
