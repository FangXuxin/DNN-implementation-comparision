import sys

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

import architecture

models_li = sorted(
    name for name in dir(architecture)
    if not (name.startswith('__') or name == 'models')
)

# define training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

cifar10 = torchvision.datasets.CIFAR10(root='Cifar10', train=False, download=True, transform=transform)

val_loader = DataLoader(cifar10, batch_size=64)

def main():

    for name in models_li:
        model = torch.load('trained_models/' + name + '.pth')
        print('validate model => {}'.format(name))
        validate(val_loader, model)

def validate(val_loader, model):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for input, target in tqdm(val_loader, desc="Test Process", file=sys.stdout):
        target = target.to(device)
        input = input.to(device)

        with torch.no_grad():
            output = model(input).view(input.size(0), -1)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print(' * Top1 acc:{top1.avg:.3f} | * Top5 acc {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n



        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()