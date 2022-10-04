# -*- coding: utf-8 -*-

"""
    @Time : 2022/10/4 20:22
    @Author : FaweksLee
    @Email : 121106010719@njust.edu.cn
    @File : AGSDNet
    @Description : 模型AGSDNet的复现(Pytorch版本)，原始代码(tensorflow):https://github.com/RTSIR/AGSDNet
"""
import numpy as np
import torch.autograd
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import interpolate

f = torch.FloatTensor(
    [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]
)
f = Variable(f)


class GCB(nn.Module):
    def __init__(self, channels):
        super(GCB, self).__init__()
        self.Gh = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, padding=1)
        self.Gh.weight = nn.Parameter(data=f, requires_grad=False)
        self.Gv = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=3, padding=1)
        self.Gv.weight = nn.Parameter(data=f, requires_grad=False)

    def forward(self, x):
        x1 = self.Gh(x)
        x2 = self.Gv(x)
        return torch.sqrt(x1 * x1 + x2 * x2)


class PAB(nn.Module):
    def __init__(self, channels):
        super(PAB, self).__init__()
        self.res = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=3, dilation=3),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.res(x)
        return torch.mul(x, x1)


class CAB(nn.Module):
    def __init__(self, channels):
        super(CAB, self).__init__()
        self.res = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=3, dilation=3),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.res(x)
        return torch.mul(x, x1)


class FDB(nn.Module):
    def __init__(self, channels):
        super(FDB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.AvgPool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 1)),
            nn.Conv2d(in_channels=65, out_channels=65, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.conv(x)
        x1_1 = self.AvgPool(x1)
        return x1 + x1_1




class AGSDNet(nn.Module):
    def __init__(self, in_channels):
        super(AGSDNet, self).__init__()
        self.GCB = GCB(in_channels)
        self.Conv1 = nn.Sequential(
            # ReLU+1-Dilated Conv
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        )
        self.Conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=65, out_channels=65, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=65, out_channels=65, kernel_size=3, dilation=3, padding=3)
        )
        self.Conv3 = FDB(65)
        self.Conv4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=65, out_channels=65, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=65, out_channels=65, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=65, out_channels=65, kernel_size=3, padding=1)
        )
        self.Conv5 = nn.Conv2d(130, 65, kernel_size=3, padding=1)
        self.PAB = PAB(65)
        self.CAB = CAB(65)
        self.Conv6 = nn.Conv2d(130, 65, kernel_size=3, padding=1)
        self.Conv7 = nn.Conv2d(65, 65, kernel_size=3, padding=1)
        self.Conv8 = nn.Conv2d(65, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.GCB(x)
        x2 = self.Conv1(x)
        x1 = torch.cat([x1, x2], dim=1)
        x2 = self.Conv2(x1)
        x2 = self.Conv3(x2)
        x3 = self.Conv4(x2)
        x3 = x1 + x3
        x3 = torch.cat([x1, x3], dim=1)
        x3 = self.Conv5(x3)
        x4_1 = self.PAB(x3)
        x4_2 = self.CAB(x3)
        x4 = torch.cat([x4_1, x4_2], dim=1)
        x4 = self.Conv6(x4)
        x4 = x1 + x4
        x4 = self.Conv7(x4)
        x4 = x4 + x1
        x4 = self.Conv8(x4)
        return x + x4


if __name__ == '__main__':
    data = torch.randn((4, 1, 512, 512)).cuda()
    model = AGSDNet(1).cuda()
    result = model(data)
    print(result)




