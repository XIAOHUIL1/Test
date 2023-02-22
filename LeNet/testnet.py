import torch
import torch.nn as nn


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            # 3*128*128 padding= (k_size-1)/2
            nn.MaxPool2d(2),  # 64*128*128---->64*64*64
            nn.ReLU()
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # 64*64*64---->128*64*64 padding = (k_size-1)/2
            nn.MaxPool2d(2),  # 64*128*128---->64*64*64
            nn.ReLU()
        )
        self.f3 = nn.Sequential(
            nn.Linear(128*32*32, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 10)
        )

def forward(self, input):
            output = self.c1(input)
            output = self.c2(output)
            output = output.view(output.size(0), -1)
            output = self.f3(output)
            return output