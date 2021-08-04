import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
from VGG.VGG19 import VGG19
from ResNet.Resnet import ResNet18

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:2" if USE_CUDA else "cpu")

transformer = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 32
epochs = 20
learning_rate = 0.01

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transformer
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transformer
)

train_dataset, _ = random_split(train_dataset, [10000, 40000])
# test_dataset, _ = random_split(test_dataset, [2000, 8000])

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

# model = VGG19()
model = ResNet18()
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(model, optimizer, criterion, train_loader):
    model.train()
    train_correct = 0.0
    total_train = 0.0
    train_loss = 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == y.data)
        total_train += y.size(0)
        train_loss += loss.item()

    epoch_loss = train_loss / total_train
    epoch_acc = train_correct.float() * 100 / total_train

    return epoch_loss, epoch_acc


def evaluate(model, criterion, test_loader):
    model.eval()
    val_correct = 0.0
    val_loss = 0.0
    total_test = 0.0
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

    val_epoch_loss = val_loss / total_test
    val_epoch_acc = val_correct.float() * 100 / total_test

    return val_epoch_loss, val_epoch_acc


for epoch in range(epochs):
    train_loss, train_acc = train(model, optimizer, criterion, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)

    print("[Epoch :  {}]".format(epoch + 1))
    print("train loss : {:.5f}, acc : {:.5f}".format(train_loss, train_acc))
    print("val loss : {:.5f}, acc : {:.5f}".format(val_loss, val_acc))
