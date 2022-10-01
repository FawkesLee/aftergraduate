# encoding: utf-8
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import numpy as np
import warnings
from sklearn import datasets
import matplotlib.pyplot as plt
warnings.warn("ignore")
# 产生环形数据
'''
n_samples 默认100，生成的总点数
shuffle:默认True
factor:内外圆之间的比例因子，默认0.8
noise：Double/None,默认None，将高斯噪声的标准差加入到数据中，数值大，取点分散
'''
x1, y1 = datasets.make_circles(
    n_samples=5000, factor=.6, noise=0.05)  # (5000,2) , (5000,)
# 产生聚类数据
x2, y2 = datasets.make_circles(n_samples=1000)  # (1000,2) , (1000,)

# 产生月牙形数据
x, y = datasets.make_moons(n_samples=1000, noise=0.1)

# 数据合并可视化
x = np.vstack((x1, x2))  # (6000,2)
y = np.hstack((y1, y2))  # (6000,)
plt.figure(figsize=(5, 5))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 绘图显示中文
plt.rcParams['axes.unicode_minus'] = False
plt.title("原始数据")
plt.scatter(x[:, 0], x[:, 1], c=y, marker='*')
plt.show()

# K-Means 分类
result = KMeans(n_clusters=3, random_state=9).fit_predict(x)
plt.figure(figsize=(5, 5))
plt.title("KMeans")
plt.scatter(x[:, 0], x[:, 1], c=result)
plt.show()

# 小批量 K-Means
result = MiniBatchKMeans(n_clusters=3, random_state=9).fit_predict(x)
plt.figure(figsize=(5, 5))
plt.title("MiniBatchKMeans")
plt.scatter(x[:, 0], x[:, 1], c=result)
plt.show()

# Birch的层次分类方法
result = Birch(n_clusters=3).fit_predict(x)
plt.figure(figsize=(5, 5))
plt.title("Birch")
plt.scatter(x[:, 0], x[:, 1], c=result)
plt.show()
