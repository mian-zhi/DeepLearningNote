## 线性回归

​	线性回归的输出是一个连续值，适用于回归问题。

​	softmax回归则适用于分类问题。

​	两者都是单层神经网络。

### 1 线性回归的基本要素

​	以房屋价格预测为例子，假设决定房屋价格（元）的因素只有：面积（平方米）和房龄（年）

#### 1.1 模型定义

​	我们设：
$$
\begin{cases}
  x_1 : Hourse\ area  
\\x_2 : Hourse\ age
\\y\ \ : Hourse\ price
\end{cases}
$$
​	则其线性关系为：
$$
\hat{y} = x_1\omega_1 + x_2\omega_2 + b
$$
​	其中：
$$
\begin{cases}
  \omega_i : weight  
\\b\ \ : bias
\end{cases}
$$
​	而 $\hat{y}$ 是预测值，允许其与真实值有一定的误差。

#### 1.2 模型训练

​	接下来我们需要通过数据来寻找特定的模型参数值，使模型在数据上的误差尽可能小。

##### 1.2.1 训练数据

​	收集一系列真实的数据，例如多栋房屋的真实价格及其对应的面积和房龄。在这个数据上面寻找模型参数来使模型的预测价格与真实价格的误差最小。称之为：训练数据集（training data set）或训练集（training set）。

​	其中：

- 一栋房屋被称为一个样本（sample）
- 真实价格称为标签（label）
- 用于预测的两个因素，被称为特征（feature）

##### 1.2.2 损失函数

​	选取一个函数用于衡量预测值与真实值的误差，称为误差函数（loss function）：

- 通常是非负数的

- 数值越小表示误差越小

​	定义如下：

$$
\varrho (\omega_1,\omega_2,b) = \frac{1}{2}(\hat{y}^{(i)}- y^{(i)})^2
$$

​	在模型训练中，我们期望找到一组模型参数，记为：$\omega_1^*,\omega_2^*,b^*$，来使训练样本平均损失最小。

##### 1.2.3 优化算法

	- 当模型和损失函数形式较为简单时，上面的误差最小化问题的解可以直接用公式表达出来。这类解叫作**解析解**（analytical solution）。
	- 然而，大多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作**数值解**（numerical solution）
	- 在求数值解的优化算法中，**小批量随机梯度下降**（mini-batch stochastic gradient descent）在深度学习中被广泛使用。

小批量随机梯度下降：

 	1. 先选取一组模型参数的初始值，如随机选取
 	2. 进行多次迭代，使每次迭代都可能降低损失函数的值
 	3. 每次迭代中，先随机均匀采样一个由固定数目训练数据样本所组成的小批量（mini-batch）$\mathcal{B}$
 	4. 然后求小批量中数据样本的平均损失有关模型参数的导数（**梯度**）
 	5. 最后用此结果与预先设定的一个正数的乘积（**步长 或 学习率**）作为模型参数在本次迭代的减小量
 	6. 在本节中，每个参数都作如下迭代

$$
\omega_1 \gets 
\omega_1 - 
\frac{\eta}{\mathcal{B}} 
\sum_{\mathcal{i}\in \mathcal{B}}^{}
\frac{\partial \varrho^{i}(\omega_1,\omega_2,b)}{\partial\omega_1} 
=
\omega_1 - 
\frac{\eta}{\mathcal{B}} 
\sum_{\mathcal{i}\in \mathcal{B}}^{}
x_1^{(i)}(x_1^{(i)}\omega_1
+x_2^{(i)}\omega_2
+b - y^{(i)}),

\\

\omega_2 \gets 
\omega_2 - 
\frac{\eta}{\mathcal{B}} 
\sum_{\mathcal{i}\in \mathcal{B}}^{}
\frac{\partial \varrho^{i}(\omega_1,\omega_2,b)}{\partial\omega_2} 
=
\omega_2 - 
\frac{\eta}{\mathcal{B}} 
\sum_{\mathcal{i}\in \mathcal{B}}^{}
x_2^{(i)}(x_1^{(i)}\omega_1
+x_2^{(i)}\omega_2
+b - y^{(i)}),

\\

b \gets 
b - 
\frac{\eta}{\mathcal{B}} 
\sum_{\mathcal{i}\in \mathcal{B}}^{}
\frac{\partial \varrho^{i}(\omega_1,\omega_2,b)}{\partial b} 
=
b \gets 
b - 
\frac{\eta}{\mathcal{B}} 
(x_1^{(i)}\omega_1
+x_2^{(i)}\omega_2
+b - y^{(i)}).
$$

	7. 其中$|\mathcal{B}|$代表每个最小批量中的样本个数（批量大小，batch size）
	8. *η* 称作学习率（learning rate）并取正数
	9. 这里的批量大小和学习率的值是人为设定的，并不是通过模型训练学出的，因此叫作超参数（hyperparameter）
	10. 通常所说的“调参”指的正是调节超参数

#### 1.3 模型预测

- 模型训练完成后，我们将模型参数$\omega_1 , \omega_2 , b$在优化算法停止时的值分别记作 $\hat{\omega_1}, \hat{\omega_1}, \hat{b}$
- 得到的并不一定是最小化损失函数的最优解$\omega_1^* , \omega_2^* , b^*$
- 通过$\hat{y} = x_1\hat\omega_1 + x_2\hat\omega_2 + \hat b$ 进行预测

### 2 从0开始线性回归

#### 2.1 生成数据集

​	我们认为生成一个数据集：

- 样本数为1000，特征数为2
- 真实的$\mathcal{\omega} = [2 , -3.4]^T$和偏差 b = 4.2
- 添加噪音项$\epsilon $ 服从均值为0、标准差为0.01的正态分布

```python
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
```

​	用绘图函数绘制：

```python
def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);
plt.show()
```

​	散点图如下：

![img](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter03/3.2_output1.png)

#### 2.2 读取数据

​	在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。我们定义一个函数，它每次返回batch_size个随机样本的特征值和标签：

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)            #读取样本大小
    indices = list(range(num_examples))     #生成一个和样本大小一样大的序列
    random.shuffle(indices)                 #打乱该序列，样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)       # 第一个参数0表示按行索引，j则是索引的行数
```

#### 2.3 初始化模型参数

​	我们把权值初始化为均值为0、标准差为0.01的正态随机数，偏差初始化为0：

```python
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)),
                 dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
```

​	之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此我们要让它们的`requires_grad=True` ：

```python
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
```

#### 2.4 定义模型

​	下面是线性回归的矢量计算表达式的实现。我们使用`mm`函数做矩阵乘法。

```python
def linreg(X, w, b): 
    return torch.mm(X, w) + b
```

#### 2.5 定义损失函数

 - 我们用平方损失来定义线性回归的损失函数
 - 在实现中，我们需要把真实值`y`变形成预测值 $\hat y$ 的形状

```python
def squared_loss(y_hat, y):  
    return (y_hat - y.view(y_hat.size())) ** 2 / 2
```

#### 2.6 定义优化算法

​	实现了上一节中介绍的小批量随机梯度下降算法

```python
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size 
        # 注意这里更改param时用的param.data
```

#### 2.7 训练模型

 	1. 在每次迭代中，我们根据当前读取的小批量数据样本（特征`X`和标签`y`）
 	2. 通过调用反向函数`backward`计算小批量随机梯度
 	3. 并调用优化算法`sgd`迭代模型参数

注意事项：

- 由于我们之前设批量大小`batch_size`为10，每个小批量的损失`l`的形状为(10, 1)。而变量`l`并不是一个标量。所以我们可以调用`.sum()`将其求和得到一个标量，再运行`l.backward()`得到该变量有关模型参数的梯度。
- 注意在每次更新完参数后不要忘了将参数的梯度清零。

```python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。
    # X和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()                     # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)      # 使用小批量随机梯度下降迭代模型参数
        w.grad.data.zero_()              # 不要忘了梯度清零 
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
    
""""
output:
    epoch 1, loss 0.028127
    epoch 2, loss 0.000095
    epoch 3, loss 0.000050
```

 - 在一个迭代周期（epoch）中，我们将完整遍历一遍`data_iter`函数，并对训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。
 - 这里的迭代周期个数`num_epochs`和学习率`lr`都是超参数，分别设3和0.03。在实践中，大多超参数都需要通过反复试错来不断调节。

```python
print(true_w, '\n', w)
print(true_b, '\n', b)

""""
output:
    [2, -3.4] 
    tensor([[1.9998],[-3.3998]], requires_grad=True)
    4.2 
    tensor([4.2001], requires_grad=True)
```

### 3 线性回归的简洁实现

#### 3.1 生成数据集

​	我们生成与上一节中相同的数据集。其中`features`是训练数据特征，`labels`是标签。

```python
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
```

#### 3.2 读取数据

​	PyTorch提供了`data`包来读取数据。由于`data`常用作变量名，我们将导入的`data`模块用`Data`代替。在每一次迭代中，我们将随机读取包含10个数据样本的小批量。

```python
import torch.utils.data as Data

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
```

#### 3.3 定义模型

​	其实，PyTorch提供了大量预定义的层，这使我们只需关注使用哪些层来构造模型。

- 导入`torch.nn`模块
- 最常见的做法是继承`nn.Module`
- 撰写自己的网络/层
- 一个`nn.Module`实例应该包含一些层以及返回输出的前向传播（forward）方法

```python
import torch.nn as nn

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
```

​	或者，使用：`nn.Sequential`来更加方便地搭建网络，`Sequential`是一个有序的容器，网络层将按照在传入`Sequential`的顺序依次被添加到计算图中。

```python
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])

""""
Sequential(
  (linear): Linear(in_features=2, out_features=1, bias=True)
)
Linear(in_features=2, out_features=1, bias=True)
```

#### 3.4 定义模型

	- PyTorch在`init`模块中提供了多种参数初始化方法。
	- 这里的`init`是`initializer`的缩写形式。
	- 我们通过`init.normal_`将权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布。偏差会初始化为零。

```python
from torch.nn import init

init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)  
# 也可以直接修改bias的data: net[0].bias.data.fill_(0)
```

- 如果这里的`net`是用3.3.3节一开始的代码自定义的，那么上面代码会报错，`net[0].weight`应改为`net.linear.weight`，`bias`亦然。
- 因为`net[0]`这样根据下标访问子模块的写法只有当`net`是个`ModuleList`或者`Sequential`实例时才可以.

#### 3.5 定义损失函数

​	PyTorch在`nn`模块中提供了各种损失函数，这些损失函数可看作是一种特殊的层，PyTorch也将这些损失函数实现为`nn.Module`的子类。

```python
loss = nn.MSELoss()
```

#### 3.6 定义优化算法

	- 我们也无须自己实现小批量随机梯度下降算法。
	- `torch.optim`模块提供了很多常用的优化算法比如SGD、Adam和RMSProp等。
	- 下面我们创建一个用于优化`net`所有参数的优化器实例，并指定学习率为0.03的小批量随机梯度下降（SGD）为优化算法。

```python
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

"""" output"
SGD (
Parameter Group 0
    dampening: 0
    lr: 0.03
    momentum: 0
    nesterov: False
    weight_decay: 0
)
```

#### 3.7训练模型

在使用Gluon训练模型时，我们通过调用`optim`实例的`step`函数来迭代模型参数。按照小批量随机梯度下降的定义，我们在`step`函数中指明批量大小，从而对批量中样本梯度求平均。

```python
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
    
"""" output:
epoch 1, loss: 0.000457
epoch 2, loss: 0.000081
epoch 3, loss: 0.000198
```

​	下面我们分别比较学到的模型参数和真实的模型参数。我们从`net`获得需要的层，并访问其权重（`weight`）和偏差（`bias`）。学到的参数和真实的参数很接近。

```python
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)

"""" output:
[2, -3.4] tensor([[ 1.9999, -3.4005]])
4.2 tensor([4.2011])
```



随便改一改



