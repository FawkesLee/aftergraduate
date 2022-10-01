# 导言
&ensp;&ensp;Non-Maximum Suppression（NMS）非极大值抑制，也有个别地方称之为非最大值抑制。个人认为前者更为贴切，因为其具体实现原理是找到所有局部最大值，并抑制非局部最大值，而不是找全局最大值，后文会介绍代码。从字面意思理解，抑制那些非极大值的元素，保留极大值元素。其主要用于目标检测，目标跟踪，3D重建，数据挖掘等。

&ensp;&ensp;目前NMS常用的有标准NMS, Soft  NMS, DIOU NMS等。后续出现了新的Softer NMS，Weighted NMS等改进版。
<div style="text-align: center;">NMS</div>
标准NMS（左图1维，右图2维）算法的伪代码:

![](https://raw.githubusercontent.com/wh21118310/drawing-bed/main/image/img.png)

左边是只计算领域范围为3的算法伪代码。以目标检测为例，目标检测推理过程中会产生很多检测框（A,B,C,D,E,F等），其中很多检测框都是检测同一个目标，但最终每个目标只需要一个检测框，NMS选择那个得分最高的检测框（假设是C），再将C与剩余框计算相应的IOU值，当IOU值超过所设定的阈值（普遍设置为0.5，目标检测中常设置为0.7，仅供参考），即对超过阈值的剩余框进行抑制，抑制的做法是将剩余的检测框的得分设置为0，如此一轮过后，在所有剩余的检测框中继续寻找得分最高的，再抑制与之IOU超过阈值的框，直到最后会保留几乎没有重叠的框。这样基本可以做到每个目标只剩下一个检测框。
```python
import torch
def NMS(boxes,scores, thresholds):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    areas = (x2-x1)*(y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter/(areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=thresholds).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)
```
 根据前面对目标检测中NMS的算法描述，易得出标准NMS容易出现的几个问题：当阈值过小时，如下图所示，绿色框容易被抑制；当过大时，容易造成误检，即抑制效果不明显。因此，出现升级版soft NMS。
 <div style="text-align: center;">Soft NMS</div>
Soft NMS算法伪代码如下：

![Soft NMS](D:\coding\Python\AfterGraduate\日常\图片素材集合\img_1.png)

在标准NMS中，IoU超过阈值的检测框的得分直接设置为0，而soft NMS主张将其得分进行惩罚衰减，有两种衰减方式，第一种惩罚函数如下:

![Soft NMS第一种惩罚方式](D:\coding\Python\AfterGraduate\日常\图片素材集合\img_2.png)

这种方式使用1-Iou与得分的乘积作为衰减后的值，但这种方式在略低于阈值和略高于阈值的部分，经过惩罚衰减函数后，很容易导致得分排序的顺序打乱，合理的惩罚函数应该是具有高iou的有高的惩罚，低iou的有低的惩罚，它们中间应该是逐渐过渡的。因此提出第二种高斯惩罚函数，具体如下：

![](D:\coding\Python\AfterGraduate\日常\图片素材集合\img_3.png)
这样Soft NMS可避免阈值设置大小的问题。Soft NMS还有后续改进版Softer-NMS，其主要解决的问题是：当所有候选框都不够精确时该如何选择，当得分高的候选框并不更精确，更精确的候选框得分并不是最高时怎么选择 。
此外，针对这一阈值设置问题而提出的方式还有Weighted NMS和Adaptive NMS。 Weighted NMS主要是对坐标进行加权平均， Adaptive NMS在目标分布稀疏时使用小阈值，保证尽可能多地去除冗余框，在目标分布密集时采用大阈值，避免漏检。
<div style="text-align: center;">DIoU</div>
当IoU相同,当相邻框的中心点越靠近当前最大得分框的中心点，则可认为其更有可能是冗余框。第一种相比于第三种更不太可能是冗余框。因此，研究者使用所提出的DIoU替代IoU作为NMS的评判准则。其定义为$DIoU=IoU-\frac{d^2}{c^2}$,其中d和c的定义如下:

![c和d](D:\coding\Python\AfterGraduate\日常\图片素材集合\img_4.png)
在DIoU中还引入参数$\beta$，用于控制对距离的惩罚程度:$DIoU=IoU-(\frac{d^2}{c^2})^\beta$。
* 当 β趋向于无穷大时，DIoU退化为IoU，此时的DIoU-NMS与标准NMS效果相当。
* 当 β趋向于0时，此时几乎所有中心点与得分最大的框的中心点不重合的框都被保留了。