import torch.nn as nn
import torch
from architecture.models.conv.amend_conv import New_conv7

class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # self.conv1 = New_conv3(in_channel,out_channel,s=stride,p=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = New_conv3(out_channel, out_channel, s=1, p=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        # self.max = nn.MaxPool2d(kernel_size=1,stride=2)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = self.max(out)
        out += identity
        out = self.relu(out)

        return out


class ResNet34_add(nn.Module):

    def __init__(self, block=BasicBlock, blocks_num=[3, 4, 6, 3], out_num=10, init_weights=True):
        super(ResNet34_add, self).__init__()
        self.in_channel = 64

        # self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = New_conv7(3, self.in_channel, s=2, p=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, blocks_num[0])
        self.layer2 = self.make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self.make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self.make_layer(block, 512, blocks_num[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(512 , out_num)
        if init_weights:
            self._initialize_weights()

    def make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel))

        layers = []
        layers.append(block(self.in_channel,channel,downsample=downsample,stride=stride))
        self.in_channel = channel

        for i in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   # 64*16*16
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 64*8*8

        x = self.layer1(x)  # 64*8*8
        x = self.layer2(x)  # 128*4*4
        x = self.layer3(x)  # 256*282
        x = self.layer4(x)  # 512*1*1


        x = self.avgpool(x) # 512*1*1
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
