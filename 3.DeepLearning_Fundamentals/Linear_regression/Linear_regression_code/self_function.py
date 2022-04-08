import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.nn
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

    # 本函数已保存在d2lzh包中方便以后使用
def data_iter(batch_size, features, labels):
    num_examples = len(features)                  #读取样本大小
    indices = list(range(num_examples))           #生成一个和样本大小一样大的序列
    random.shuffle(indices)                       #打乱该序列，样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)       # 第一个参数0表示按行索引，j则是索引的行数

def linreg(X, w, b):         # 前向计算公式
    return torch.mm(X, w) + b

def squared_loss(y_hat, y):   # 损失函数
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):  # 优化算法
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

