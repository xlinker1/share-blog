---
draft: false
tags:
  - todo
---

## softmax
- softmax中包含指数计算，为了数值稳定性，会将所有输入减去最大值
- safe softmax包含三次循环：1. 找最大值 2. 算指数，求和来得到分母 3. 除以分母
- online softmax将前两次合并到一起。每次求和时使用的是当时的局部最大值，如果发现了新的最大值会乘一个系数进行调整。最后一次循环不变
- 

求导推导
$y_i$是对应位置$x_i$ 在做完softmax后的结果。
$y_i=\frac{e^{x_i}}{\sum_{j=1}^ne^{x_j}}$
先算$\frac{\partial y_i}{\partial x_j}$，当i=j时：
$$
\begin{align}
\frac{\partial y_i}{\partial x_j}&=\frac{\partial y_i}{\partial x_i} \\
&=\frac{\partial \exp(\log(y_i))}{\partial \log(y_i)}\frac{\partial \log(y_i)}{\partial x_i} \\
&=y_i \frac{\partial (x_i - \log(\sum_{j=1}^ne^{x_j}))}{\partial x_i} \\
&=y_i (1-\frac{\partial \log(\sum_{j=1}^ne^{x_j})}{\partial x_i}) \\
&=y_i (1-\frac{e^{x_i}}{\sum_{j=1}^ne^{x_j}}) \\
&=y_i (1-y_i) \\
\end{align}
$$
当i!=j时：
$$
\begin{align}
\frac{\partial y_i}{\partial x_j}&=\frac{\partial y_i}{\partial x_j} \\
&=\frac{\partial \exp(\log(y_i))}{\partial \log(y_i)}\frac{\partial \log(y_i)}{\partial x_j} \\
&=y_i \frac{\partial (x_i - \log(\sum_{k=1}^ne^{x_k}))}{\partial x_j} \\
&=y_i (-\frac{\partial \log(\sum_{k=1}^ne^{x_k})}{\partial x_j}) \\
&=y_i (-\frac{e^{x_j}}{\sum_{k=1}^ne^{x_k}}) \\
&=- y_i y_j \\
\end{align}
$$

最后$\frac{\partial l}{\partial x_j}=\sum_i \frac{\partial l}{\partial y_i}\frac{\partial y_i}{\partial x_j}=\sum_i \frac{\partial l}{\partial y_i} y_i (-y_j)+\frac{\partial l}{\partial y_j} y_j=y_j(\frac{\partial l}{\partial y_j}-\sum_i \frac{\partial l}{\partial y_i} y_i)$
看起来稍微有点乱，大概就是反向传回来的梯度对softmax的值做一个加权求和$s=\sum_i \frac{\partial l}{\partial y_i} y_i$，再在每个位置算$y_j(\frac{\partial l}{\partial y_j}-s)$

附加一个对-log_softmax求导的推导，这个会用来算交叉熵 [[llm.c]]

求导推导
$y_i$是对应位置$x_i$ 在做完softmax后的结果。
$y_i=\frac{e^{x_i}}{\sum_{j=1}^ne^{x_j}}$，而$z_i=-\log(y_i)$
先算$\frac{\partial z_i}{\partial x_j}$，当i=j时：
$$
\begin{align}
\frac{\partial z_i}{\partial x_j}&=\frac{\partial z_i}{\partial x_i} \\
&=-\frac{\partial \log(y_i)}{\partial x_i} \\
&=-\frac{\partial (x_i - \log(\sum_{j=1}^ne^{x_j}))}{\partial x_i} \\
&=(\frac{\partial \log(\sum_{j=1}^ne^{x_j})-x_i}{\partial x_i}) \\
&=\frac{e^{x_i}}{\sum_{j=1}^ne^{x_j}}-1 \\
&=y_i-1 \\
\end{align}
$$
当i!=j时：
$$
\begin{align}
\frac{\partial z_i}{\partial x_j}&=-\frac{\partial \log(y_i)}{\partial x_j} \\
&=-\frac{\partial (x_i - \log(\sum_{j=1}^ne^{x_j}))}{\partial x_j} \\
&=(\frac{\partial \log(\sum_{j=1}^ne^{x_j})-x_i}{\partial x_j}) \\
&=\frac{e^{x_j}}{\sum_{j=1}^ne^{x_j}} \\
&=y_j \\
\end{align}
$$


## Attention

[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
这篇文章里有标准attention的前向计算和反向计算怎么算的，还有它提出的优化。反向计算和一些细节在附录里。

## 标准self-attention

前向传播公式：
$$\begin{aligned}&\mathbf{S}=\tau\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{S}^{\mathrm{masked}}=\mathrm{MASK}(S)\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S}^{\mathrm{masked}})\in\mathbb{R}^{N\times N},\\&\mathbf{P}^{\mathrm{dropped}}=\mathrm{dropout}(\mathbf{P},p_{\mathrm{drop}}),\quad\mathbf{O}=\mathbf{P}^{\mathrm{dropped}}\mathbf{V}\in\mathbb{R}^{N\times d},\end{aligned}$$

标准attention的反向传播
![[Pasted image 20240525101046.png]]

问自己几个问题：需要保存多少中间值？需要保存注意力矩阵吗？重计算更好吗？mask需要存吗？ #todo 

### FlashAttention v1
前向
反向

### FlashAttention v2

[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://tridao.me/publications/flash2/flash2.pdf)



### ring attention
[ring attention + flash attention：超长上下文之路](https://zhuanlan.zhihu.com/p/683714620)
[大模型训练之序列并行双雄：DeepSpeed Ulysses & Ring-Attention](https://zhuanlan.zhihu.com/p/689067888)




------
大模型推理加速技术的学习路线是什么? - 猛猿的回答 - 知乎
https://www.zhihu.com/question/591646269/answer/3309904882

vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现 - 方佳瑞的文章 - 知乎
https://zhuanlan.zhihu.com/p/673284781

【手撕LLM-Flash Attention】从softmax说起，保姆级超长文！！ - 小冬瓜AIGC的文章 - 知乎
https://zhuanlan.zhihu.com/p/663932651

[【手撕LLM-FlashAttention2】只因For循环优化的太美 - 知乎](https://zhuanlan.zhihu.com/p/670085985)


