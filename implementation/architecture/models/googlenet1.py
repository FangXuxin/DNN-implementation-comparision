import torch.nn as nn
import torch
import torch.nn.functional as F
# from architecture import *
from architecture.models.conv.amend_conv import New_conv5

class GoogLeNet_add(nn.Module):
    def __init__(self, out_num=10, aux_logits=True, init_weights=True):
        super(GoogLeNet_add, self).__init__()
        self.aux_logits = aux_logits

        # self.conv1 = BasicConv2d(3, 64, kernel_size=3, stride=2, padding=3) # change
        self.conv1 = New_conv3(3, 64, 3, 2)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        # self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = New_conv3(64, 192, p=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=3, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, out_num)
            self.aux2 = InceptionAux(528, out_num)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, out_num)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)  # output => 16*16*64
        x = self.maxpool1(x)    # output => 8*8*64
        x = self.conv2(x)   # output => 8*8*64
        x = self.conv3(x)   # output => 8*8*192
        x = self.maxpool2(x)    # output => 4*4*192
        x = self.inception3a(x) # output => 4*4*256
        x = self.inception3b(x) # output => 4*4*480
        x = self.maxpool3(x)     # output => 2*2*480
        x = self.inception4a(x) # output => 2*2*512

        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x) # output => 2*2*512
        x = self.inception4c(x) # output => 2*2*512
        x = self.inception4d(x) # output => 2*2*528

        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)     # output => 2*2*832
        x = self.maxpool4(x)    # output => 1*1*832
        x = self.inception5a(x) # output => 1*1*832
        x = self.inception5b(x) # output => 1*1*832
        x = self.avgpool(x) # output => 1*1*1024
        x = torch.flatten(x, 1)  # output => 1024
        x = self.dropout(x)
        x = self.fc(x)  # 10

        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
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

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            # BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
            New_conv3(ch3x3red, ch3x3, p=1) # Ensure that the output size is equal to the input size
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            # BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
            New_conv5(ch5x5red, ch5x5, p=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, out_num):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=1, stride=3)    # change
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, out_num)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x




class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class New_conv3(nn.Module):
    def __init__(self, in_ch, out_ch, p=1, s=1, **kwargs):
        super(New_conv3, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), padding=(0,p), stride=s, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), padding=(p,0), stride=s, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x