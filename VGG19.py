import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:2' if USE_CUDA else 'cpu')

transformer = transforms.Compose([transforms.Resize((224,224)),
                                  transforms.ToTensor()
                                  ])

batch_size = 32
epochs = 20
learning_rate = 0.01

train_dataset = datasets.CIFAR10(root='./data',train=True,download=True,transform=transformer)
test_dataset = datasets.CIFAR10(root='./data',train=False,download=True,transform=transformer)

train_dataset, _ = random_split(train_dataset, [10000, 40000])
test_dataset, _ = random_split(test_dataset, [2000, 8000])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)

        self.conv5 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv6 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv7 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv8 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)

        self.conv9 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv10 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv11 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv12 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)

        self.conv13 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv14 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv15 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv16 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)

        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.pool(x)

        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return F.softmax(x)

model = VGG19().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train_loss = 0.0
    train_correct = 0.0
    val_loss = 0.0
    val_correct = 0.0
    total_train = 0.0
    total_test = 0.0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == y.data)
        total_train += y.size(0)
        train_loss += loss.item()
        #if i%50==0:
        #    print("{:} : loss = {:.5f}".format(i, train_loss/(i+1)))

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == y.data)
            total_test += y.size(0)
            val_loss += loss.item()

    epoch_loss = train_loss / total_train
    epoch_acc = train_correct.float()*100 / total_train

    val_epoch_loss = val_loss / total_test
    val_epoch_acc = val_correct.float()*100 / total_test

    print("[Epoch :  {}]".format(epoch+1))
    print("train loss : {:.5f}, acc : {:.5f}".format(epoch_loss, epoch_acc))
    print("val loss : {:.5f}, acc : {:.5f}".format(val_epoch_loss, val_epoch_acc))