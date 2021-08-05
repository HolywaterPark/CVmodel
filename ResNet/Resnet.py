from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    mul = 1
    def __init__(self, insize, outsize, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            insize, outsize, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(outsize)
        self.conv2 = nn.Conv2d(
            outsize, outsize, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(outsize)
        self.shortcut = nn.Sequential()
        if stride != 1 or insize != outsize:
            self.shortcut = nn.Sequential(
                nn.Conv2d(insize, outsize, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outsize),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    mul = 4
    def __init__(self, insize, outsize, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            insize, outsize, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(outsize)
        self.conv2 = nn.Conv2d(
            outsize, outsize, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(outsize)
        self.conv3 = nn.Conv2d(outsize, self.mul*outsize, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.mul*outsize)

        self.shortcut = nn.Sequential()
        if stride != 1 or insize != self.mul*outsize:
            self.shortcut = nn.Sequential(
                nn.Conv2d(insize, self.mul*outsize, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.mul*outsize),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.insize = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block, num_blocks[0], 64, stride=1)
        self.layer2 = self.make_layer(block, num_blocks[1], 128, stride=2)
        self.layer3 = self.make_layer(block, num_blocks[2], 256, stride=2)
        self.layer4 = self.make_layer(block, num_blocks[3], 512, stride=2)
        self.fc = nn.Linear(512*block.mul, num_classes)

    def make_layer(self, block, num_blocks, outsize, stride):
        layers = []
        for num_block in range(num_blocks - 1):
            layers.append(block(self.insize, outsize))
            self.insize = outsize * block.mul
        layers.append(block(self.insize, outsize, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])