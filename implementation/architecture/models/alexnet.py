import torch.nn as nn
import torch

class Alexnet(nn.Module):
    def __init__(self, out_num=10, init_weights=True):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=2),    # con1 => 96*34*34
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # Maxpool1 => 16*16*96
            nn.Conv2d(96, 256, kernel_size=5, padding=2, stride=2),    # con2 => 8*8*256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # Maxpool2 => 3*3*256
            nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=1),    # con3 => 3*3*384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1),  # con4 => 3*3*384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1),  # con5 => 3*3*256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Maxpool3 => 1*1*256
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),   # => 256*1*1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_num)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        x = torch.flatten(x, start_dim=1)
        # print(x.size())
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