import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
import typer

from model.VGG import VGG19
from model.Resnet import ResNet18, ResNet34, ResNet50
from model.DCGAN import Generator, Discriminator

app = typer.Typer()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:2" if USE_CUDA else "cpu")


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


def weights_init_GAN(model):
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def DCGAN(train_loader):
    epochs = 3
    learning_rate = 0.0002

    modelG = Generator().to(device)
    modelD = Discriminator().to(device)

    modelG.apply(weights_init_GAN)
    modelD.apply(weights_init_GAN)

    criterion = nn.BCELoss().to(device)
    optimizerG = torch.optim.Adam(
        modelG.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    optimizerD = torch.optim.Adam(
        modelD.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )

    G_loss_list = []
    D_loss_list = []

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = torch.full((x.size(0),), 1, dtype=torch.float, device=device)

            modelD.zero_grad()
            outputs = modelD(x).view(-1)
            loss_real = criterion(outputs, y)
            loss_real.backward()

            noise_vec = torch.randn(x.size(0), 100, 1, 1, device=device)
            fake_data = modelG(noise_vec)
            y.fill_(0)
            outputs = modelD(fake_data.detach()).view(-1)
            loss_fake = criterion(outputs, y)
            loss_fake.backward()

            loss_D = loss_real + loss_fake
            optimizerD.step()

            modelG.zero_grad()
            y.fill_(1)
            outputs = modelD(fake_data).view(-1)
            loss_G = criterion(outputs, y)
            loss_G.backward()
            optimizerG.step()

            G_loss_list.append(loss_G.item())
            D_loss_list.append(loss_D.item())
            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t"
                    % (
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss_D.item(),
                        loss_G.item(),
                    )
                )


def cifar_10(batch_size, transformer):
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transformer
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transformer
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


@app.command("train_test")
def train_func_test(dataset_name: str, neuralnet_name: str):
    batch_size = 128
    epochs = 300
    learning_rate = 0.01

    if neuralnet_name == "VGG19":
        model = VGG19()
        transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif neuralnet_name == "ResNet18":
        model = ResNet18()
        transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif neuralnet_name == "ResNet34":
        model = ResNet34()
        transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif neuralnet_name == "ResNet50":
        model = ResNet50()
        transformer = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif neuralnet_name == "DCGAN":
        transformer = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )

    if dataset_name == "cifar-10":
        train_loader, test_loader = cifar_10(batch_size, transformer)

    if neuralnet_name == "DCGAN":
        DCGAN(train_loader)
        return

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        train_loss, train_acc = train(model, optimizer, criterion, train_loader)
        val_loss, val_acc = evaluate(model, criterion, test_loader)
        train_acc_list.append(train_acc.cpu().numpy().astype(np.float32))
        val_acc_list.append(val_acc.cpu().numpy().astype(np.float32))

        print("[Epoch :  {}]".format(epoch + 1))
        print("train loss : {:.5f}, acc : {:.5f}".format(train_loss, train_acc))
        print("val loss : {:.5f}, acc : {:.5f}".format(val_loss, val_acc))

    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.show()
