---
draft: false
tags:
---
[1412.6980v8.pdf](https://arxiv.org/pdf/1412.6980v8.pdf)
伪代码如下：
![[Pasted image 20240124100448.png]]
反向传播后，每个可训练参数都有对应的梯度。我们需要对梯度$\nabla_\theta f(\theta_t)$在时间维度上计算指数平均，来估计梯度下降的平均方向$u$。我们需要对梯度的平方$(\nabla_\theta f(\theta_t))^2$在时间维度上计算指数平均，来估计梯度下降在各个参数维度上的平均步长$v^{1/2}$。

估计出来当前时刻的梯度下降平均方向和平均步长后，我们就用$u_{t+1}/v_{t+1}^{1/2}$来对每个参数进行更新。用平均方向$u$更新参数的好处是，减少前后几次梯度更新的抖动情况。除以平均步长$v_{t+1}^{1/2}$的好处是，让更新梯度一直很小的参数，在每次更新时可以有更大的更新量，让更新梯度一直很大的参数，有更小的更新量，从而让训练更稳定。

我自己一开始看的时候，比较迷惑的是这里的bias correction到底是怎么算的。下面给一个解释
下面是通过迭代的方式计算$g_t$的指数平均，$m_t$表示从开始训练到时刻t的所有梯度的指数平均。$g_t$从t=1开始，$m_0=0$
$m_t=\beta\cdot m_{t-1}+(1-\beta)\cdot g_t$
展开得到
$m_t=(1-\beta)\cdot g_t +(1-\beta)\beta\cdot g_{t-1}+(1-\beta)\beta^2\cdot g_{t-2}+\dots+(1-\beta)\beta^{t-1}g_1$
我们是希望它在对过去各个时候的梯度做加权和，希望权重之和为1。但是上面这个式子的权重和不等于1

所以只要对$m_t$除以权重和就得到了归一化的$\hat m_t$
$$
\begin{aligned}
\hat{m}_t&= \frac{(1-\beta)\cdot g_t +(1-\beta)\beta\cdot g_{t-1}+(1-\beta)\beta^2\cdot g_{t-2}+\dots+(1-\beta)\beta^{t-1}g_1}{(1-\beta) +(1-\beta)\beta+(1-\beta)\beta^2+\dots+(1-\beta)\beta^{t-1}} \\
&=\frac{g_t+\beta g_{t-1}+\dots+\beta^{t-2}g_2+\beta^{t-1}g_1}{1+\beta+\dots+\beta^{t-2}+\beta^{t-1}} \\
\end{aligned}
$$
权重和等于
$$
\begin{aligned}
& (1-\beta) +(1-\beta)\beta+(1-\beta)\beta^2+\dots+(1-\beta)\beta^{t-1} \\
=& (1-\beta)(1+\beta+\dots+\beta^{t-1}) \\
=& (1-\beta) \frac{1-\beta^t}{1-\beta}\\
=& 1-\beta^t
\end{aligned}
$$

所以 $\hat m_t = m_t / (1-\beta^t)$


- $m_t$为一阶动量，代表惯性，当前梯度的更新方向不仅要考虑当前梯度，还要考虑历史梯度。不希望被当前单个更新方向影响太大。
- $v_t$为二阶动量，用于控制自适应学习率，在分母位置，其物理意义在于：
	- 对于经常更新的参数，不希望被单个样本影响太大，希望学习率慢一点
	- 对于偶尔更新的参数，希望能从少量的样本中多学一些，即学习率大一些


[\[PDF\] Fixing Weight Decay Regularization in Adam | Semantic Scholar](https://www.semanticscholar.org/paper/Fixing-Weight-Decay-Regularization-in-Adam-Loshchilov-Hutter/45dfef0cc1ed96558c1c650432ce39d6a1050b6a)

在sgd中，对权重的L2正则化和weight decay是一样的，因此可以通过在损失里加一个权重L2正则项来限制权重，防止过拟合。在loss里对权重$w_i$加L2正则化，在求导后，相当于对每个权重$w_i$的梯度增加了一项$\beta*w_i$，这种做法是weight decay，目的都是让权重趋向于0。
但是在adam等自适应梯度下降算法里，在loss里加L2正则化和weight decay是不同的。当把L2范数直接作用在损失函数上时，L2范数会参与动量和方差的估计（下图中粉色部分，图中的$w_t$是weight decay系数，$x_t$才是参数），而在AdamW里，在实际更新参数时才加入weight decay（下图绿色部分）
实验证明AdamW的做法效果更好，想约束权重就直接约束，不要在loss里间接约束。

![[Pasted image 20240524204455.png]]
图中”学习率“有$\eta_t$和$\alpha$，注意它们和weight decay系数的嵌套关系
（在[[llm.c]]的实现里没有$\alpha$）
（其实也可以把weight decay系数$w_tx_{t-1}$移到动量$m_t$旁边，比如$(m_t+w_tx_{t-1})/\sqrt{v_t}$ 这种，不知道有没有人试过）


[Weight Decay的前世今生和隐藏缺陷](https://zhuanlan.zhihu.com/p/672650395)


