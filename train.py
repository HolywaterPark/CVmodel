import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import typer

from VGG.VGG import VGG11, VGG13, VGG16, VGG19
from ResNet.Resnet import ResNet18, ResNet34, ResNet50
from DCGAN import Generator, Discriminator

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@app.command("cifar10")
def cifar_10(n:str):
    transformer = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 128
    epochs = 300
    learning_rate = 0.01

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

    if n == "VGG19":
        model = VGG19()
    elif n == "ResNet18":
        model = ResNet18()
    elif n == "ResNet34":
        model = ResNet34()
    elif n == "ResNet50":
        model = ResNet50()


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
    plt.savefig('ResNet50.png')

@app.command("DCGAN")
def DCGAN():
    transformer = transforms.Compose(
        [transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 32
    epochs = 200
    learning_rate = 0.0002

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

    modelG = Generator().to(device)
    modelD = Discriminator().to(device)

    modelG.apply(weights_init)
    modelD.apply(weights_init)

    criterion = nn.BCELoss().to(device)
    optimizerG = torch.optim.Adam(modelG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(modelD.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    G_loss_list = []
    D_loss_list = []

    for x,y in train_loader:
        ll = x.size(0)
        break

    test_noise = torch.randn(ll, 100, 1, 1, device=device)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            #### Discriminator
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

            #### Generator
            modelG.zero_grad()
            y.fill_(1)
            outputs = modelD(fake_data).view(-1)
            loss_G = criterion(outputs, y)
            loss_G.backward()
            optimizerG.step()

            G_loss_list.append(loss_G.item())
            D_loss_list.append(loss_D.item())
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                      % (epoch, epochs, i, len(train_loader), loss_D.item(), loss_G.item()))

        if epoch % 20 == 0:
            with torch.no_grad():
                fake_test = modelG(test_noise)
                plt.imshow(np.transpose(make_grid(fake_test.cpu())))
                plt.savefig('epoch{:}.png'.format(epoch+1))


    val_loss, val_acc = evaluate(modelG, criterion, test_loader)
    print('val_loss : {:.5f} | val_acc : {:.5f}'.format(val_loss, val_acc))

    plt.plot(G_loss_list, label="G")
    plt.plot(D_loss_list, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig('DCGAN.png')

