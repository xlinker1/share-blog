---
draft: false
tags:
---




DPO论文 [link](https://arxiv.org/pdf/2305.18290.pdf)

首先看看RLHF一般是怎样训练的：
- 训练奖励函数 $r_{\phi}$
	- $\min_{r_{\phi}}\mathbb{E}_{(x,y_\text{win},y_\text{lose})\sim\mathcal{D}}[-\log\sigma(r_\phi(x,y_\text{win})-r_\phi(x,y_\text{lose}))]$
	- 人类的偏好由下面的概率$p^*$来描述，最小化上面这个目标函数得到的最优的$r_\phi$就可以表示人的偏好。
	- $$ \begin{aligned}
	  p^*(y_\text{win} \succ y_\text{lose}) &= \frac{\exp(r^*(x,y_\text{win}))}{\exp(r^*(x,y_\text{win}))+\exp(r^*(x,y_\text{lose}))} \\
	  &= \sigma(r^*(x,y_\text{win})-r^*(x,y_\text{lose}))
	  \end{aligned}
	  $$
	- 因为$p^*$的大小只取决于奖励之间的差值，所以奖励的绝对值并不重要
	  $r(x,y)$和$r(x,y)-f(x)$都表示同一个人类偏好$p^*$。在作为奖励训练语言模型时，往往把训练出来的奖励值用某种方式给归一化。

- 在强化学习阶段，需要调整策略$\pi_\theta$最大化奖励。同时约束策略不要偏离参考策略$\pi_\text{ref}$太远
	- $max_{\pi_\theta} \mathbb{E}_{x\sim D,y\sim\pi_\theta(y|x)}[r_\phi(x,y)] - \beta \mathbb{E}_{x\sim D}[\mathbb{D}_{KL}[\pi_\theta(y|x)||\pi_\text{ref}(y|x)]]$
	- 而$\mathbb{D}_{KL}[\pi_\theta(y|x)||\pi_\text{ref}(y|x)]=\mathbb{E}_{y\sim\pi_\theta(y|x)}[\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}]$，可以合并到前一项去
	- 如果将后面的KL散度约束和前一项合并，也可以说是采用了新的奖励函数$R(x,y)=r_\phi(x,y)-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}$. 那么目标函数就是$max_{\pi_\theta} \mathbb{E}_{x\sim D,y\sim\pi_\theta(y|x)}[R(x,y)]$
	- 这个最大化奖励也可以写成最大化一个KL散度的形式，从而求出解析解

上面强化学习阶段最大化的目标函数其实可以求出解析解。也就是说，给定表示人类偏好的奖励函数$r_\phi$和参考模型$\pi_\text{ref}$以及正则化系数$\beta$，我们可以求出使上面那个表达式最大的$\pi^*$
$$
\begin{align}
& \mathbb{E}_{x\sim D,y\sim\pi_\theta(y|x)}[r_\phi(x,y)] - \beta \mathbb{E}_{x\sim D}[\mathbb{D}_{KL}[\pi_\theta(y|x)||\pi_\text{ref}(y|x)]] \\
=& \mathbb{E}_{x\sim D,y\sim\pi_\theta(y|x)}[r_\phi(x,y)-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}] \\
=& \mathbb{E}_{x\sim D,y\sim\pi_\theta(y|x)}[-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)e^{\frac{r_\phi(x,y)}{\beta}}}}]  \tag{1}
\end{align} 

$$
注意到KL散度的表达式$\mathbb{D}_{KL}[\pi_\theta(y|x)||\pi_\text{ref}(y|x)]=\mathbb{E}_{y\sim\pi_\theta(y|x)}[\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}]$，形式上和上面的式(1)非常像。而且当$\pi_\theta=\pi_\text{ref}$时KL散度取最小值为零。于是就考虑能不能把式(1)中log里的分母改成概率分布的形式。
令$\pi_r(y|x)=\frac{\pi_\text{ref}(y|x)e^{\frac{r_\phi(x,y)}{\beta}}}{Z(x)},~Z(x)=\sum_y{\pi_\text{ref}(y|x)e^{\frac{r_\phi(x,y)}{\beta}}}$
式(1)就变成：$\mathbb{E}_{x\sim D,y\sim\pi_\theta(y|x)}[-\beta\log{\frac{\pi_\theta(y|x)}{\pi_r(x|y)}}]+\beta\mathbb{E}_{x\sim D}Z(x)$
想要通过调整$\pi_\theta(y|x)$最大化上式，其实就是要最小化$\mathbb{E}_{x\sim D,y\sim\pi_\theta(y|x)}[\log{\frac{\pi_\theta(y|x)}{\pi_r(x|y)}}]$
只需要让$\pi_\theta=\pi_r$即可
也就是说对每一个奖励函数和参考模型约束，我们都可以找到强化学习后的最优策略，也就是$\pi_r(x|y)=\frac{\pi_\text{ref}(y|x)e^{\frac{r_\phi(x,y)}{\beta}}}{Z(x)}$
反过来说，每一个策略$\pi_\theta$都可以对应于一个奖励函数$r_\phi$，其中$Z(x)$在后面代入目标函数时可以被抵消掉
$r_\phi(x,y)/\beta=\log{\frac{\pi_r(y|x)}{\pi_\text{ref}(y|x)}}+\log{Z(x)}$
将奖励函数的表达式代入训练奖励函数时需要最小化的目标函数
$$
\begin{align}
&\mathbb{E}_{(x,y_\text{win},y_\text{lose})\sim\mathcal{D}}[-\log\sigma(r_\phi(x,y_\text{win})-r_\phi(x,y_\text{lose}))] \\
=& \mathbb{E}_{(x,y_\text{win},y_\text{lose})\sim\mathcal{D}}[-\log\sigma(\beta\log{\frac{\pi_\theta(y_\text{win}|x)}{\pi_\text{ref}(y_\text{win}|x)}}-\beta\log{\frac{\pi_\theta(y_\text{lose}|x)}{\pi_\text{ref}(y_\text{lose}|x)}})]
\end{align}
$$
然后我们再对$\theta$求梯度得到：
$$
-\beta \mathbb{E}_{(x,y_\text{win},y_\text{lose})}[\sigma(\hat{r}_\theta(x,y_\text{lose})-\hat{r}_\theta(x,y_\text{win}))(\nabla_\theta\log\pi_\theta(y_\text{win}|x) - \nabla_\theta\log\pi_\theta(y_\text{lose}|x))]
$$
其中$\hat{r}_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$，是当前策略$\pi_\theta$对应的奖励函数。
它和一般RLHF的区别就在于，没有直接训练奖励函数，而是用当前策略和参考策略之间的差值表示待优化的奖励函数。所以原论文中说“Your Language Model Is Secretly a Reward Model”，但是这样的Reward Model好不好呢？在人类偏好数据上的泛化性如何呢？训练出来的策略效果如何呢？因为虽然表达式一样，但是先训练奖励模型，后训练策略，和两个一起训，毕竟是不一样的。

我们再来看看梯度的表达式，DPO到底在做什么？
后一项是在说，增加好输出的概率，减少坏输出的概率，和SFT差不多。前一项是说，如果$\hat{r}_\theta(x,y)$能够足够好的分辨出好输出和坏输出，那么就不用在这份数据上做梯度更新了。

### 使用dpo拟合reward model
先显式的训练一个奖励模型是有好处的，训练好之后可以利用它的泛化性降低对标注数据的需求。所以可以先训练一个奖励模型，然后让dpo的奖励模型去拟合实际训练的奖励模型。

有两种思路，一种是直接拟合奖励函数的值。这种方法的问题主要是Z(x)不知道，要额外调一个模型来估计。
$r_\phi(x,y)/\beta=\log{\frac{\pi_r(y|x)}{\pi_\text{ref}(y|x)}}+\log{Z(x)}$
$Z(x)=\sum_y{\pi_\text{ref}(y|x)e^{\frac{r_\phi(x,y)}{\beta}}}$


怎么看怎么麻烦，因此我自己更偏向于下面这种思路。

另一种思路是对dpo进行改良，使用交叉熵来让dpo奖励模型输出的赢率拟合真正奖励模型算出的赢率。令$\sigma_w=\sigma(r(x,y_w)-r(x,y_l))$
$$
\begin{align}
&\mathbb{E}_{(x,y_\text{w},y_\text{l})\sim\mathcal{D}}[-\sigma(r(x,y_w)-r(x,y_l))\log\sigma(r_\phi(x,y_\text{w})-r_\phi(x,y_\text{l}))-(1-\sigma(r(x,y_w)-r(x,y_l)))\log\sigma(r_\phi(x,y_\text{l})-r_\phi(x,y_\text{w}))] \\
=&\mathbb{E}_{(x,y_\text{w},y_\text{l})\sim\mathcal{D}}[-\sigma_w \log\sigma(r_\phi(x,y_\text{w})-r_\phi(x,y_\text{l}))-(1-\sigma_w )\log\sigma(r_\phi(x,y_\text{l})-r_\phi(x,y_\text{w}))] 
\end{align}
$$

对$\theta$求梯度得到：
$$
\begin{align}
& -\beta \mathbb{E}_{(x,y_\text{w},y_\text{l})}[\sigma_w (1-\sigma(\hat{r}_\theta(x,y_\text{w})-\hat{r}_\theta(x,y_\text{l}))(\nabla_\theta\log\pi_\theta(y_\text{w}|x) - \nabla_\theta\log\pi_\theta(y_\text{l}|x))+
(1-\sigma_w) \sigma(\hat{r}_\theta(x,y_\text{w})-\hat{r}_\theta(x,y_\text{l}))(-\nabla_\theta\log\pi_\theta(y_\text{w}|x) +\nabla_\theta\log\pi_\theta(y_\text{l}|x))] \\
=& -\beta \mathbb{E}_{(x,y_\text{w},y_\text{l})}[(\sigma_w -\sigma(\hat{r}_\theta(x,y_\text{w})-\hat{r}_\theta(x,y_\text{l}))(\nabla_\theta\log\pi_\theta(y_\text{w}|x) - \nabla_\theta\log\pi_\theta(y_\text{l}|x))
\end{align}
$$

当$\sigma_w=1$时，梯度表达式变为原始dpo的梯度。



#todo

### 讨论

dpo的一些缺点：
1. 和普通的rlhf一样，奖励模型在序列末尾给奖励，属于output supervise，因此同样存在奖励分配(credit assignment)问题。即每个生成的token对最终奖励的贡献有多大？
	1. rlhf通过在ppo阶段使用启发式的方法给每个token分配奖励/计算优势函数。dpo也有可能做类似的改进，具体来说就是给不同token反向传播的梯度通过推导，设置不同的权重。相关工作有：token dpo
2. dpo没有显式训练奖励模型，因此无法利用奖励模型本身的泛化性。而对于预训练+微调的范式来说，泛化性才是最重要的。改进方法也很简单，就是显式的训练一个最好的奖励模型，然后让dpo的奖励模型目标函数去拟合它。可以是直接拟合奖励模型的输出打分，也可以是用交叉熵拟合奖励模型判断出来的两个序列的赢率。
3. 



直接训练奖励模型的必要性：
1. 不同的奖励模型，训练出来的效果不同，奖励模型本身的泛化性能也不同。甚至可以尝试使用奖励模型的集成来提升奖励质量。
2. 可以利用正在训练的模型$\pi_\theta$来生成回复对，然后依据自己生成的结果进行奖励。
3. 训练了一个奖励模型，可以用奖励模型的泛化性来生成人类喜好标注。然后来利用额外的未标注的数据
4. 可以通过$\pi_\text{ref}$来生成样本，利用奖励模型当作系数来拒绝采样。来生成奖励模型对应的$\pi_\text{opt}$样本。因为$\pi_\text{opt}(y|x)\propto \pi_\text{ref}(y|x)e^{r(x,y)/\beta}$



和在末尾进行奖励的rlhf一样，dpo也存在奖励分配的问题。
Q：是否可以不是对整个response做dpo，而是逐个token，或者逐个step做dpo呢？
A：应该不能。dpo要求在相同前缀x下，对两个不同的回应y1,y2做二分类。如果不是对结果奖励，而是对中间步骤奖励，要对每个中间步骤都找到正例和负例，让训练的数据形成一种树形结构才能训练。
能否用其它方式来做奖励分配呢？我觉得很难，最多也就是用启发式的方法加权之类的。
所以我觉得dpo最好的用法，可能还是去帮助llm拟合output reward model。

另外之所以需要用$\min_{r_{\phi}}\mathbb{E}_{(x,y_\text{win},y_\text{lose})\sim\mathcal{D}}[-\log\sigma(r_\phi(x,y_\text{win})-r_\phi(x,y_\text{lose}))]$训练奖励模型，是因为人类偏好没有绝对答案，人与人之间的偏好也不完全一样，只能一对一对的比较。但如果是数学问题的话，每一步正确与否是有客观标准的，输出多分类的标签就好了。



