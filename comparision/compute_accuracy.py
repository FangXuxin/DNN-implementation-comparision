import json
import sys
import time

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import topk

# define training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
transform = torchvision.transforms.Compose(
    [

        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ]
)



imagenet = torchvision.datasets.ImageFolder(root='Imagenet/val', transform=transform)


# use dataloader
test_loader = DataLoader(imagenet, batch_size=64)


def load_model(name):
    path = 'models\\' + name + '.pth'
    model = torch.load(path)
    model.to(device)
    return model


def validate(val_loader, model):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for input, target in tqdm(val_loader, desc="Test Process", file=sys.stdout):
        target = target.to(device)
        input = input.to(device)

        with torch.no_grad():
            output = model(input).view(input.size(0), -1)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

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


def main():
    f = open('models\\name.txt', 'r')
    js = f.read()
    model_li = json.loads(js)
    f.close()

    acc_dict = {}

    for name in model_li:
        # setting
        temp_acc = []

        # create model
        model = load_model(name)

        # evaluate model
        print('=> Use pretrained model ' + name)
        top1_acc, top5_acc = validate(test_loader, model, name)
        temp_acc.append(top1_acc)
        temp_acc.append(top5_acc)
        acc_dict[name] = temp_acc

    print(acc_dict)

    js = json.dumps(acc_dict)
    with open('result/accuracy.txt', 'w') as f:
        f.write(js)
    f.close()
    print('done!')





if __name__ == '__main__':
    main()

