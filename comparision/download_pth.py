import json

import torch
import torchvision.models as models

models_dic = {
    'alexnet' : models.alexnet(pretrained=True),
    'vgg11' : models.vgg11(pretrained=True),
    'vgg13' : models.vgg13(pretrained=True),
    'vgg16' : models.vgg16(pretrained=True),
    'vgg19' : models.vgg19(pretrained=True),
    'googlenet' : models.googlenet(pretrained=True),
    'resnet18' :models.resnet18(pretrained=True),
    'resnet34' : models.resnet34(pretrained=True),
    'resnet50' : models.resnet50(pretrained=True),
    'resnet101' : models.resnet101(pretrained=True),
    'resnet152' : models.resnet152(pretrained=True),
}

name_li = []

for name, model in models_dic.items():
    name_li.append(name)
    torch.save(model, 'models\\' + name + '.pth')
    print('=> Model {} has saved!'.format(name))

with open('models\\name.txt', 'w') as f:
    name_li.sort()
    js = json.dumps(name_li)
    f.write(js)
    f.close()



