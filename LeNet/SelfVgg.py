import torch
from torch import nn


# VGG分为三部分
class SelfVgg16Net(nn.Module):
    def __init__(self):
        super(SelfVgg16Net, self).__init__()
        # first block
        self.c1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # second block
        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # third block
        self.c5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # fourth block
        self.c8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.c9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.c10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # fifth block
        self.c11 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.c12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.p5 = nn.MaxPool2d(kernel_size=2, stride=2)
