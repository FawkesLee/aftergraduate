# encoding:utf-8
# @Author: DorisFawkes
# @File:
# @Date: 2021/09/02 17:13
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

data = np.arange(64).reshape((8, 8))
A = torch.Tensor(data.reshape(1, 1, 8, 8))
print("A=", A)

## MAX POOL
maxpool = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
B, indices = maxpool(A)
print("MaxPooling:")
print(B.shape)
print('B=', B)

# 双线性插值上采样实现方法一， scale_factor=n指的是宽高的尺寸放大n倍
Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
C = Upsample(B)
print("UpSample:")
print(C.shape)
print('C=', C)

# 双线性插值上采样实现方法二， scale_factor=n指的是宽高的尺寸放大n倍
D = F.interpolate(B, scale_factor=2, mode='bilinear')
print("Interpolate:")
print(D.shape)
print('D=', D)

### max unpool
maxunpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
E = maxunpool(B, indices)
print("MaxPooling:")
print(E.shape)
print('E=', E)
