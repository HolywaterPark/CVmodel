from torch import nn
from torch.nn import functional as F


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )

        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )

        self.conv9 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv11 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv12 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )

        self.conv13 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv14 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv15 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )
        self.conv16 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
        )

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool3(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool4(x)

        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return F.softmax(x)
