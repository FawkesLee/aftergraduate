import cv2
import numpy as np
import matplotlib.pyplot as plt


# 求两个点的差值
def getGrayDiff(image, currentPoint, tmpPoint):
    return abs(int(image[currentPoint[0], currentPoint[1]]) - int(image[tmpPoint[0], tmpPoint[1]]))


# 区域生长算法
def regional_growth(gray, seeds, threshold=5):
    # 每次区域生长的时候的像素之间的八个邻接点
    connects = [(-1, -1), (0, -1), (1, -1), (1, 0),
                (1, 1), (0, 1), (-1, 1), (-1, 0)]
    # threshold为生长时候的相似性阈值，默认即灰度级不相差超过15以内的都算为相同
    height, weight = gray.shape
    seedMark = np.zeros(gray.shape)
    seedList = []
    for seed in seeds:
        if(seed[0] < gray.shape[0] and seed[1] < gray.shape[1] and seed[0] > 0 and seed[1] > 0):
            seedList.append(seed)  # 将添加到的列表中
    print(seedList)
    label = 1  # 标记点的flag
    while(len(seedList) > 0):  # 如果列表里还存在点
        currentPoint = seedList.pop(0)  # 将最前面的那个抛出
        seedMark[currentPoint[0], currentPoint[1]] = label  # 将对应位置的点标志为1
        for i in range(8):  # 对这个点周围的8个点一次进行相似性判断
            tmpX = currentPoint[0] + connects[i][0]
            tmpY = currentPoint[1] + connects[i][1]
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:  # 如果超出限定的阈值范围
                continue  # 跳过并继续
            grayDiff = getGrayDiff(
                gray, currentPoint, (tmpX, tmpY))  # 计算此点与像素点的灰度级之差
            if grayDiff < threshold and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append((tmpX, tmpY))
    return seedMark

# 初始种子选择


def originalSeed(gray):
    # 二值图，种子区域(不同划分可获得不同种子)
    ret, img1 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    """cv2.threshold(src,thresh,maxval,type[,dst]) -> retval, dst
        src表示图像源,thresh表示阈值(起始值),maxval表最大值,type表示划分使用的算法,常用值为0，即 cv2.THRESH_BINARY
    """
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img1)  # 进行连通域操作，取其质点
    centroids = centroids.astype(int)  # 转化为整数
    return centroids


originimg = cv2.imread("./data/640.png")
originimg = cv2.cvtColor(originimg, cv2.COLOR_BGR2GRAY)
img = originimg.copy()
seed = originalSeed(img)
img = regional_growth(img, seed)
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 图像显示
plt.figure(figsize=(10, 5))  # width * height
plt.subplot(121), plt.imshow(
    originimg, cmap='gray'), plt.title("原始图像"), plt.axis("off")
plt.subplot(122), plt.imshow(img, cmap='gray'), plt.title(
    '区域生长以后'), plt.axis("off")
plt.show()
print("ok...")
