---
draft: false
tags:
---

[\[1506.02438\] High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
策略梯度算法中，很重要的一块就是估计优势函数A(s,a)，它表示的是一个动作究竟有多好。有了它，我们才能训练策略函数去增大好动作的概率，降低坏动作的概率。下面按照自己的理解推导一下，和原论文有些不同。

##  价值函数估计

先回顾价值函数的定义 $V(s_t)=\mathbb{E}_{a_t,s_{t+1},\dots|s_t}\left[r_t+r_{t+1}+\cdots \right]$
Q函数的定义 $Q(s_t,a_t)=\mathbb{E}_{s_{t+1},\dots|s_t,a_t}\left[r_t+r_{t+1}+\cdots\right]$
其中$r_t=r_t(s_t,a_t)$，${a_t,s_{t+1},\dots|s_t}$ 表示的是给定当前状态$s_t$，后续使用$\pi(a_t|s_t)$和$e(s_{t+1}|s_t,a_t)$采样出来的轨迹。${s_{t+1},\dots|s_t,a_t}$表示的是给定当前状态和动作$s_t,a_t$，后续用策略和环境展开的轨迹序列。
将V函数的定义拆开，可以发现
$$\begin{aligned}
V(s_t)&=\mathbb{E}_{a_t,s_{t+1},\dots|s_t} \left[r_t+r_{t+1}+\cdots \right] \\
&=\mathbb{E}_{a_t,s_{t+1}|s_t}\mathbb{E}_{a_{t+1},s_{t+2},\dots|s_{t+1}} \left[r_t+r_{t+1}+\cdots \right] \\
&=\mathbb{E}_{a_t,s_{t+1}|s_t}\left[ r_t +\mathbb{E}_{a_{t+1},s_{t+2},\dots|s_{t+1}} \left[r_{t+1}+\cdots \right] \right]\\
&=\mathbb{E}_{a_t,s_{t+1}|s_t}\left[ r_t +V(s_{t+1})\right]\\
\end{aligned}$$
一般来说我们在使用actor-critic类算法时，需要训练critic，也就是用神经网络参数化的V函数$V_\theta(s_t)$。上面这个式子说明，可以用轨迹里面的 $r_t+r_{t+1}+\cdots$ 作为训练目标，它是对$V(s_t)$的无偏估计，但是因为产生轨迹时要做很多次采样，所以方差大。也可以用$r_t +V_\theta(s_{t+1})$作为训练目标，它的方差小，但因为$V_\theta(s_{t+1})$不准而存在偏差。

对比上面两个训练目标，通过使用同样的对轨迹展开的方法，可以得出一种折中的估计目标
$V(s_t)=\mathbb{E}_{a_t,s_{t+1}\cdots s_{t+n}|s_t}\left[ r_t +\cdots+r_{t+n-1} +V(s_{t+n})\right]$
它的方差变大了，但是偏差....欸还是一样！

为了对它进行改进，我们修改价值函数的定义。让价值函数不是当前状态往后所获得的所有平均奖励，而是当前状态往后所获得的所有**折扣**平均奖励，即太远的奖励，我就在当前价值函数里忽略掉。这样做一方面可以忽略太久之后的奖励（反正1. 估计不准 2. 大概率和当前动作关系不大），另一方面大多数时候从当前状态玩一局游戏到结束获得的奖励的和不一定为有限值，这种时候就必须对奖励做折扣。
此时带折扣$\gamma$的价值函数定义如下：
$V(s_t)=\mathbb{E}_{a_t,s_{t+1},\dots|s_t} \left[ r_t+\gamma r_{t+1}+\gamma^2 r_{t+2} +\cdots \right]$
估计目标如下：
$V(s_t)=\mathbb{E}_{a_t,s_{t+1}\cdots s_{t+n}|s_t}\left[ r_t +\cdots+\gamma^{n-1} r_{t+n-1} +\gamma^n V(s_{t+n})\right]$

这样我们就可以通过多往后展开几步来降低$V_\theta(s_{t+1})$估计不准带来的偏差。但同时中间因为多展开了几步也增加了方差。
上面有很多个估计目标，虽然估计的是同一个东西，但是有的偏差高有的方差高。所以我们还可以对它们做指数加权平均，来在方差和偏差之间权衡。
令

使用$\lambda$对它们加权平均：

#todo
（这里暂时跳过了，因为可以从下面推的Adv1估计中得到）

##  优势函数估计

优势函数的定义是：$A(s_t,a_t)=Q(s_t,a_t)-V(s_t)$
一般来说我们都是有一个用神经网络训练的价值函数$V_\theta(s_t)$，采样得到很多轨迹和奖励，然后试图估计$A(s_t,a_t)$。

仔细看Q函数的定义 $Q(s_t,a_t)=\mathbb{E}_{s_{t+1},\dots|s_t,a_t}\left[r_t+r_{t+1}+\cdots\right]$，它和价值函数的区别在于，价值函数的$a_t$通过在$\pi$上采样得来，Q函数的$a_t$是确定的。
$V(s_t)=\mathbb{E}_{a_t,s_{t+1},\dots|s_t}\left[r_t+r_{t+1}+\cdots \right]$
把Q函数展开
$$\begin{aligned}
Q(s_t,a_t)&=\mathbb{E}_{s_{t+1},\dots|s_t,a_t}\left[r_t+r_{t+1}+\cdots\right] \\

&=\mathbb{E}_{s_{t+1}|s_t,a_t}\mathbb{E}_{a_{t+1},s_{t+2},\dots|s_{t+1}} \left[r_t+r_{t+1}+\cdots \right] \\
&=\mathbb{E}_{s_{t+1}|s_t,a_t}\left[ r_t +\mathbb{E}_{a_{t+1},s_{t+2},\dots|s_{t+1}} \left[r_{t+1}+\cdots \right] \right]\\
&= r_t +\mathbb{E}_{s_{t+1}|s_t,a_t}\left[V(s_{t+1})\right]\\
\end{aligned}$$

这样我们只要能准确的估计价值函数，就可以比较准的估计优势函数了。
$A(s_t,a_t)=r_t +\mathbb{E}_{s_{t+1}|s_t,a_t}\left[V(s_{t+1})\right]-V(s_t)$
在这个估计式子里，$V_\theta(s_t)$会带来偏差，但在策略梯度里面我们证明过，系数加减$b(s_t)$不会影响梯度估计的偏差。所以主要会带来偏差的是$V(s_{t+1})$。我们可以采用和上面价值函数估计类似的做法，对奖励做折扣，然后对价值函数估计的表达式展开，最后对各个估计式子做指数加权，就可以得到GAE。

如果奖励带折扣，那么优势函数为
$A(s_t,a_t)=r_t +\mathbb{E}_{s_{t+1}|s_t,a_t}\left[ \gamma V(s_{t+1})\right]-V(s_t)$
光是看着表达式可能有点晕，实际上在计算的时候直接忽略那个期望，用轨迹上的奖励$r_t$和$V(s_t),V(s_{t+1})$来计算，就是下面这个数。

$$\begin{aligned}
\hat A_t^{(1)}&=-V(s_t)+r_t +\gamma V(s_{t+1})
\end{aligned}$$
期望里的东西和价值估计一样，可以往后拆开，拆开之后再忽略期望，即：
$$\begin{aligned}
\hat A_t^{(1)}&=-V(s_t)+r_t +\gamma V(s_{t+1}) \\
\hat A_t^{(2)}&=-V(s_t)+r_t +\gamma r_{t+1}+\gamma^2 V(s_{t+2}) \\
\hat{A}_{t}^{(3)}&=-V(s_{t})+r_{t}+\gamma r_{t+1}+\gamma^{2}r_{t+2}+\gamma^{3}V(s_{t+3}) \\
\hat{A}_{t}^{(k)}&=-V(s_{t})+r_{t}+\gamma r_{t+1}+\cdots+\gamma^{k-1}r_{t+k-1}+\gamma^{k}V(s_{t+k}) 
\end{aligned}$$

和价值估计一样，计算$\hat A_t^{(n)}$时，越是往后展开，r越多，方差越大，$\gamma^n V(s_{t+n})$越小，偏差越小。因此也可以对它们做指数加权平均，来在方差和偏差之间权衡。

使用$\lambda$对它们加权平均：
$$\begin{aligned}
\hat{A}_t^{\mathrm{GAE}}&=(1-\lambda)\left(\hat{A}_t^{(1)}+\lambda\hat{A}_t^{(2)}+\lambda^2\hat{A}_t^{(3)}+\ldots\right) \\
&=(1-\lambda)\left(
\sum_{i=0}^\infty \lambda^i \hat{A}_t^{(i+1)} \right) \\
&=(1-\lambda)\left(
-\sum_{i=0}^\infty \lambda^i V(s_t)+\sum_{i=0}^\infty \lambda^i \gamma^{i+1} V(s_{t+i+1})+
\sum_{i=0}^\infty\lambda^i\sum_{k=0}^{i}\gamma^k r_{t+k}
\right) \\

&=-V(s_t)+(1-\lambda)\sum_{i=0}^\infty \lambda^i \gamma^{i+1} V(s_{t+i+1})+(1-\lambda) \sum_{i=0}^\infty\lambda^i\sum_{k=0}^{i}\gamma^k r_{t+k} \\
&=-V(s_t)+(1-\lambda)\sum_{i=0}^\infty \lambda^i \gamma^{i+1} V(s_{t+i+1})+(1-\lambda) \sum_{k=0}^{\infty}\sum_{i=k}^\infty\lambda^i\gamma^k r_{t+k} \\
&=-V(s_t)+(1-\lambda)\sum_{i=0}^\infty \lambda^i \gamma^{i+1} V(s_{t+i+1})+(1-\lambda) \sum_{k=0}^{\infty} \frac{\lambda^k}{1-\lambda} \gamma^k r_{t+k} \\
&=-V(s_t)+(1-\lambda)\sum_{i=0}^\infty \lambda^i \gamma^{i+1} V(s_{t+i+1})+ \sum_{k=0}^{\infty} \lambda^k \gamma^k r_{t+k} \\
\end{aligned}$$
（推的累人...尤其是那个求和换序....还好离散数学里讲过，又有gpt帮忙）


最后在代码里长这样，Adv1指的是$\hat{A}_1^{\mathrm{GAE}}$，是$A(s_1,a_1)$的估计
（毛了一段别人的注释过来）
```
Advantages looks like this:
Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
	  - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

Returns looks like this:
Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
		   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...
```

Ret1=Adv1+V1 ，是$Q(s_1,a_1)$的估计。也可以作为目标训练$V_\theta(s_1)$ （我是这么理解的）



#### 再回顾一下两个参数的物理意义
$\gamma$是折扣系数，它削弱远处的奖励的影响。如果原本是想对没有打折扣的优势函数进行估计，那么它会引入偏差（在估计梯度的时候）。
$V(s_t)=\mathbb{E}_{a_t,s_{t+1},\dots|s_t} \left[ r_t+\gamma r_{t+1}+\gamma^2 r_{t+2} +\cdots \right]$


$\lambda$是对不同的估计进行加权。$\lambda$大，就相当于把估计时的轨迹拉的很长，以减少$V_\theta(s_t)$的偏差。
$\lambda$小，就相当于用$\hat A_t^{(1)}=-V(s_t)+r_t +\gamma V(s_{t+1})$估计，轨迹很短，方差很小，但是$V_\theta(s_t)$引入的偏差很大。
将$\lambda=1$代入上面推导的式子，得到，$\hat{A}_t^{\mathrm{GAE}}=\sum_{i=0}^\infty\gamma^i r_{t+i}-V(s_t)$
将$\lambda=0$代入上面的式子，得到，$\hat{A}_t^{\mathrm{GAE}}=\hat A_t^{(1)}$

