import torch
import torch.nn as nn
from architecture.models.conv.amend_conv import New_conv3


class VGG16_add(nn.Module):
    def __init__(self, out_num=10, init_weights=True):
        super(VGG16_add, self).__init__()
        self.features = nn.Sequential(
            # output => 32*32*64
            New_conv3(3,64,1,1),
            New_conv3(64, 64, 1, 1),

            # output => 16*16*64
            nn.MaxPool2d(kernel_size=2, stride=2),

            # output => 16*16*128
            New_conv3(64, 128, 1, 1),
            New_conv3(128, 128, 1, 1),

            # output => 8*8*128
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            # output => 8*8*256
            New_conv3(128, 256, 1, 1),
            New_conv3(256, 256, 1, 1),
            New_conv3(256, 256, 1, 1),

            # output => 4*4*256
            nn.MaxPool2d(kernel_size=2, stride=2),

            # output => 4*4*512
            New_conv3(256, 512, 1, 1),
            New_conv3(512, 512, 1, 1),
            New_conv3(512, 512, 1, 1),

            # output => 2*2*512
            nn.MaxPool2d(kernel_size=2, stride=2),

            # output => 2*2*512
            New_conv3(512, 512, 1, 1),
            New_conv3(512, 512, 1, 1),
            New_conv3(512, 512, 1, 1),

            # output => 1*1*512
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.features = nn.Sequential(
        #     # output => 32*32*64
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #
        #     # output => 16*16*64
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #
        #     # output => 16*16*128
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #
        #     # output => 8*8*128
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     #
        #     # output => 8*8*256
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #
        #     # output => 4*4*256
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #
        #     # output => 4*4*512
        #     nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #
        #     # output => 2*2*512
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #
        #     # output => 2*2*512
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #
        #     # output => 1*1*512
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, out_num),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
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
