import torch.nn as nn
import torch

class New_conv3(nn.Module):
    def __init__(self, in_ch, out_ch, p=1, s=1, **kwargs):
        super(New_conv3, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), padding=p, stride=s, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), padding=p, stride=s, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class New_conv5(nn.Module):
    def __init__(self, in_ch, out_ch, p=1, s=1, **kwargs):
        super(New_conv5, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=p, stride=s, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=p, stride=s, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class New_conv7(nn.Module):
    def __init__(self, in_ch, out_ch, p=1, s=1, **kwargs):
        super(New_conv7, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=p, stride=s, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=p, stride=s, **kwargs)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=p, stride=s, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        return x