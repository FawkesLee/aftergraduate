# -*- coding: utf-8 -*-

"""
@Time : 2022/5/7
@Author : FaweksLee
@File : test
@Description : 
"""
import numpy as np
import torch
# from einops import rearrange
# import torch.nn.functional as F
# a = torch.randn(2, 3, 3, 3)
# print(a)
# # b = a.mean(3)  # size: 2, 3, 3
# # c = b.mean(2)  # size: 2, 3
# b = torch.mean(a, 2, True)
# # c = torch.mean(a, 2)
# d = torch.mean(b, 3, True)
# e = F.interpolate(d, size=(64, 64), scale_factor=None, mode='bilinear', align_corners=True)
# print(b)
# # print(c)
data = torch.randint(0, 9, size=(3, 3))
data = data.unsqueeze(0)
print(data, "\n")
data1 = torch.max(data, dim=2)
print(data1, "\n")
data2 = torch.max(data, dim=1)[0]
print(data2)
