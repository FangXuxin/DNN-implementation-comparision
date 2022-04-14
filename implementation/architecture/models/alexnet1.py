import torch.nn as nn
import torch
from architecture.models.conv.amend_conv import New_conv3,New_conv5


class Alexnet_add(nn.Module):
    def __init__(self, out_num=10, init_weights=True):
        super(Alexnet_add, self).__init__()
        self.conv1 = New_conv3(3,96,2,1)
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = New_conv5(96,256, 2, 2)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = New_conv3(256,384,1,1)
        self.conv4 = New_conv3(384, 384, 1, 1)
        self.conv5 = New_conv3(384, 256, 1, 1)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*3*3, 4096),   # => 256*1*1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_num)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max3(x)

        # x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
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

