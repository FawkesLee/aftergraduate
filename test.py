import torch
a = torch.randn((4, 2, 2, 2, 2, 2))
b = torch.randn((4, 2, 2, 2, 2, 2))
c = list()
for i in range(0, a.size()[0]):
    for j in range(0, a.size()[1]):
        e = a[i, j, ...]
    #     print("e:", e.size())
    # f = torch.cat([d, e], dim=0)
    # c.append(f)
# print(c.size())
