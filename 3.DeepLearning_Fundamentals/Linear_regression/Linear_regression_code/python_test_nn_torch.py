from ctypes import sizeof
from pickletools import optimize
from tkinter import N
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
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
dataset = Data.TensorDataset(features, labels)                  #将训练数据的特征和标签组合
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)  #随机读取小批量
for X, y in data_iter:
    print(X, y)
    break

class LinearNet(nn.Module):
    
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net) # 使用print可以打印出网络的结构

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)
loss = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr = 0.03)
print(optimizer)

num_epochs = 10
for epoch in range(1 , num_epochs + 1):
    for X , y in data_iter:
        output = net(X)
        l = loss(output , y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

dense = net.linear
print(true_w, dense.weight)
print(true_b, dense.bias)
