import sys
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

import architecture
from test import validate

models_li = sorted(
    name for name in dir(architecture)
    if not (name.startswith('__') or name == 'models')
)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# for dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

train_cifar10 = torchvision.datasets.CIFAR10(root='Cifar10', train=True,
                                       download=True, transform=transform)
val_cifar10 = torchvision.datasets.CIFAR10(root='Cifar10', train=False,
                                       download=True, transform=transform)

train_loader = DataLoader(train_cifar10, batch_size=64)
val_loader = DataLoader(val_cifar10, batch_size=64)


def train(model, epoch, name):
    writer = SummaryWriter('logs/' + name)

    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters() ,lr=learning_rate)


    train_step = 0; step = 0

    for i in range(epoch):
        model.train()
        print('=> Start {} training session.'.format(i+1))

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            if name == 'GoogLeNet':
                output, aux1, aux2 = model(imgs)
                loss0 = loss_fn(output, targets)
                loss1 = loss_fn(aux1, targets)
                loss2 = loss_fn(aux2, targets)
                loss = loss0 + loss1*0.3 + loss2*0.3
            else:
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)

            writer.add_scalar(name + " Loss", loss, step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0:
                print('Loss => {}'.format(loss))


        top1, top5 = validate(val_loader, model)

        writer.add_scalar(name + " Top1", top1, train_step)
        writer.add_scalar(name + " Top5", top5, train_step)

        print('Finish {} training session.'.format(train_step+1))
        train_step += 1


def main():
    for name in models_li:
        model = architecture.__dict__[name](out_num = 10)
        model.to(device)
        epoch = 100
        print("start training architecture => {}".format(name))
        train(model, epoch, name)
        torch.save(model,'trained_models/' + name + '.pth')



if __name__ == '__main__':
    main()