from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.convt1 = nn.ConvTranspose2d(100, 512, kernel_size=4, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.convt2 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(256)

        self.convt3 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(128)

        self.convt4 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(64)

        self.convt5 = nn.ConvTranspose2d(
            64, 3, kernel_size=4, stride=2, padding=1, bias=False
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.convt1(x)), True)
        x = F.relu(self.bn2(self.convt2(x)), True)
        x = F.relu(self.bn3(self.convt3(x)), True)
        x = F.relu(self.bn4(self.convt4(x)), True)
        x = self.convt5(x)
        return F.tanh(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = self.conv5(x)
        return F.sigmoid(x)
