# encoding:utf-8
# @Author: DorisFawkes
# @File:
# @Date: 2021/08/13 20:52
import numpy as np
import torch
from torchvision.ops import deform_conv2d
from torch import nn


class net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(net, self).__init__()
        self.SimpleConv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                    padding=1)
        self.conv_offset = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        init_offset = torch.Tensor(np.zeros([18, 3, 3, 3]))
        self.conv_offset.weight = nn.Parameter(init_offset)
        self.conv_mask = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1)
        init_mask = torch.Tensor(np.zeros([9, 3, 3, 3]) + np.array([0.5]))
        self.conv_mask.weight = nn.Parameter(init_mask)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # out = self.relu(self.SimpleConv(x))
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))
        out = deform_conv2d(input=x, offset=offset, weight=self.SimpleConv.weight, mask=mask, padding=(1, 1))
        return out


if __name__ == '__main__':
    data = torch.randn((4, 3, 512, 512))
    model = net(3, 64)
    print(model(data).size())  # 4,64,512,512