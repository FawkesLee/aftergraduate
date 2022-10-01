# encoding:utf-8
# @Author: DorisFawkes
# @File:
# @Date: 2021/09/29 8:11
# ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
# 对于二维数组，可使用.flags查看数组是行(C)连续还是列(Fortran)连续
import numpy as np

arr = np.arange(12).reshape(3, 4)
flags = arr.flags
print("", arr)
print(flags)
'''
C_CONTIGUOUS : True，行连续
F_CONTIGUOUS : False,列不连续
进行arr.T或arr.transpose(1,0)是列连续行不连续
若对进行切割操作，会改变连续性，对行切割，变为行列均不连续.
'''
arr = np.arange(12).reshape(3, 4)
arr1 = arr[:, 0:2]
flags = arr1.flags
print("", arr1)
print(flags)
arr2 = arr[0:2, :]
flags = arr2.flags
print("", arr2)
print(flags)
