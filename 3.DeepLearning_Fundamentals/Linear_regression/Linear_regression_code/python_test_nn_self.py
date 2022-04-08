from ctypes import sizeof
from tkinter import N
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
sys.path.append("..")
from self_function import *

num_inputs = 2                     #number of inputs
num_examples = 1000                #number of examples

true_w = [2, -3.4]                 #the ture weight
true_b = 4.2                       #the ture bias
features = torch.randn(num_examples, num_inputs,       #Randomly generate a 1000 by 2 vector
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b      #true expression
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),          #Add noise that follows a normal distribution
                       dtype=torch.float32)
set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
plt.show()

batch_size = 10
for x , y in data_iter(batch_size, features, labels):
    print(x, y)
    break

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True) 

lr = 0.03
num_epochs = 5
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()                     # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)      # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)
