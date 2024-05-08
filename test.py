import numpy as np
import torch
import timm
import re
from torchsummary import summary

# 模型列表
# print(len(timm.models.list_models()))
# print(timm.models.list_models())
from timm import create_model as creat

# 'swinv2_base_window12to16_192to256'
# 'vit_base_patch16_224'
# 'swin_small_patch4_window7_224'
model_ft = creat('swin_tiny_patch4_window7_224', pretrained=True, num_classes=11)

print(summary(model_ft, input_size=(3, 224, 224)))
# import torch.nn as nn
# loss_func = nn.CrossEntropyLoss()
# pre = torch.tensor([[0.8, 0.5, 0.2, 0.5],
#                     [0.2, 0.9, 0.3, 0.2],
#                     [0.4, 0.3, 0.7, 0.1],
#                     [0.1, 0.2, 0.4, 0.8]], dtype=torch.float)
# tgt = torch.tensor([0, 1, 2, 3], dtype=torch.float)
# print(loss_func(pre, tgt))

# X = torch.zeros(2, 2)
# X[0, 0] = 1
# X[1, 1] = 1
# print(X)
#
# V,D=torch.linalg.eig(X)
# print(V,D)

# import torch

# def expm(A):
#     D,V=torch.linalg.eig(A)
#     return torch.matmul(torch.matmul(V,torch.diag(torch.exp(D))),torch.inverse(V))

# def logm(A, n=100):
#     """
#     计算对称正定矩阵A的矩阵对数。
#     参数:
#     A -- 输入的对称正定矩阵
#     n -- 泰勒级数的项数，默认为100
#     返回值:
#     logA -- 计算得到的矩阵对数
#     """
#     I = torch.eye(A.size(0))  # 创建单位矩阵
#     logA = torch.zeros_like(A)  # 初始化矩阵对数为零矩阵
#
#     for k in range(1, n + 1):
#         Ak = torch.pow(A, k)  # 计算A的k次幂
#         inv_Ak = torch.inverse(Ak)  # 计算Ak的逆矩阵
#         logA += (k - 1) * inv_Ak * A / k  # 累加泰勒级数的一项
#
#     logA *= (1 / n)  # 乘以系数1/n
#     return logA

# def logm(A, n=100):
#     """
#     logm(A) = (1 / n) * sum((k - 1) * inv(A^k) * A, k = 1 to n)
#     计算对称正定矩阵A的矩阵对数。
#     参数:
#     A -- 输入的对称正定矩阵
#     n -- 泰勒级数的项数，默认为100
#     返回值:
#     logA -- 计算得到的矩阵对数
#     """
#     I = torch.eye(A.size(0))  # 创建单位矩阵
#     logA = torch.zeros_like(A)  # 初始化矩阵对数为零矩阵
#
#     for k in range(1, n + 1):
#         Ak = torch.pow(A-I, k)  # 计算A的k次幂
#         logA += Ak*(-1)**k  # 累加泰勒级数的一项
#     return logA
#
# # 示例矩阵
# A = torch.tensor([[1,2,9],[2,5,7],[9,7,3]], dtype=torch.float32)
# expA=torch.expm1(A)
# expA1=torch.exp(A)
# logA = torch.log(expA)
# print("矩阵指数为:\n", expA)
# print(expA1)
# print("矩阵对数为:\n", logA)

from torchvision import transforms

