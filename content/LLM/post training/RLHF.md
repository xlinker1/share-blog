---
draft: false
tags:
---
[[强化学习与大语言模型]]



![[diagram2x-2.png]]



# RLHF

RLHF的核心是用人类偏好数据训练奖励模型，以此来引导模型的行为。
openAI之前在这方面的工作：
- 训练奖励模型，让智能体学会在模拟环境后空翻 [link](https://openai.com/research/learning-from-human-preferences) [Ilya Sutskever MIT讲座2018：OpenAI 元学习与自我对弈](https://www.bilibili.com/video/BV1wb4y1M7iY)
- [\[1909.08593\] Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) 最早在gpt2上调的版本。这时训练奖励模型还是四个回答选一个（现在一般是二选一）
-  [\[2009.01325\] Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- [Site Unreachable](https://arxiv.org/abs/2112.09332)
- [\[2203.02155\] Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

LLM里的RLHF是用人类偏好数据训练$Q(s,a)$，然后让语言模型在问题/状态$s$下的输出回答/动作$a$。（其实也不能算Q函数，因为只做一步动作，之后就没有奖励了）

训练过程如下：
- 让模型M对问题x生成多个回答y。人对每个回答两两排序，就是简单选择自己是更喜欢回答y1还是y2
- 依据人类的偏好数据训练奖励模型R。奖励模型会尝试对每个问答对(x,y)打分，会利用sigmoid函数来估计分高的问答对(x,y1)比分低的问答对(x,y2)被人类喜好的概率是多少。（奖励模型的目标函数，类似elo评分[[强化学习笔记#elo分数]]）
- 根据奖励模型R，利用policy gradient的方法调整模型M的动作，让模型生成的结果增大奖励。
  因为是策略梯度，所以要求每一次算梯度时都使用最新的模型M'采样动作。为了能用旧模型M采样的数据多更新几次，ppo使用某种方式约束，让原模型M和当前模型M'输出的策略尽可能不要偏离太多。[[强化学习笔记#基础组成#策略梯度]]
- 奖励模型R并不是完美的，它只是在模型M的输出分布上对人类偏好的估计比较准。但是对更新多步后的模型M'的输出上对人类偏好的估计就不一定准了。因此需要在M'的输出上采样，人类标注偏好，然后更新奖励模型R'。

整个过程是on-policy的，具体是两方面：
1. 奖励函数的训练是on-policy的（不然不准）
2. 模型M用policy gradient调整输出的时候是on-policy的


## 人类偏好训练奖励函数

- 训练奖励函数 $r_{\phi}$
	- $\min_{r_{\phi}}\mathbb{E}_{(x,y_\text{win},y_\text{lose})\sim\mathcal{D}}[-\log\sigma(r_\phi(x,y_\text{win})-r_\phi(x,y_\text{lose}))]$
	- 人类的偏好由下面的概率$p^*$来描述，最小化上面这个目标函数得到的最优的$r_\phi$就可以表示人的偏好。

$$\begin{aligned}
	  p^*(y_\text{win} \succ y_\text{lose}) &= \frac{\exp(r^*(x,y_\text{win}))}{\exp(r^*(x,y_\text{win}))+\exp(r^*(x,y_\text{lose}))} \\
	  &= \sigma(r^*(x,y_\text{win})-r^*(x,y_\text{lose}))
\end{aligned}
$$

因为$p^*$的大小只取决于奖励之间的差值，所以奖励的绝对值并不重要
$r(x,y)$和$r(x,y)-f(x)$都表示同一个人类偏好$p^*$。在作为奖励训练语言模型时，往往把训练出来的奖励值用某种方式给归一化。

## 调整模型以最大化奖励函数

#### 强化学习问题设置

- 目标函数是 $max_{\pi_\theta} \mathbb{E}_{x\sim D,y\sim\pi_\theta(y|x)}[r_\phi(x,y)] - \beta \mathbb{E}_{x\sim D}[\mathbb{D}_{KL}[\pi_\theta(y|x)||\pi_\text{ref}(y|x)]]$ 
	- 其中第一项是要最大化奖励模型的值，第二部分用KL散度约束当前策略和参考策略之间的距离不要太远。
	- 而$\mathbb{D}_{KL}[\pi_\theta(y|x)||\pi_\text{ref}(y|x)]=\mathbb{E}_{y\sim\pi_\theta(y|x)}[\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}]$，可以合并到前一项去。
	- 如果将后面的KL散度约束和前一项合并，也可以说是采用了新的奖励函数$R(x,y)=r_\phi(x, y)-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}$. 那么目标函数就是$max_{\pi_\theta} \mathbb{E}_{x\sim D,y\sim\pi_\theta(y|x)}[R(x,y)]$
	- 这个最大化奖励也可以写成最大化一个KL散度的形式，从而求出解析解。（见[[DPO]]

这里的强化学习其实有几个问题：
1. 为什么是$\mathbb{E}_{x\sim D}[\mathbb{D}_{KL}[\pi_\theta(y|x)||\pi_\text{ref}(y|x)]]$ ？为什么不是$\mathbb{E}_{x\sim D}[\mathbb{D}_{KL}[\pi_\text{ref}(y|x)||\pi_\theta(y|x)]]$ 呢？ppo论文里的kl-penalty就是后者，暂时不知道有啥区别。后者还是从参考策略里采样呢。可能用前者是为了方便合并到奖励函数里吧。
2. 需要kl散度约束的原因是奖励函数不够完美，它只在$x\sim D,y\sim\pi_\theta(y|x)$的输出上表现比较好。当$\pi_\theta$被多次更新之后，奖励函数的准确性会下降。但这个约束不是强化学习也不是ppo，只是比较像。
3. 奖励函数其实不能简单的写作$R(x,y)=r_\phi(x, y)-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}$，应该是$R_\theta(x,y)=r_\phi(x, y)-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}$. 在策略梯度的推导中（见[[强化学习笔记#直接优化策略函数（Policy Gradient）]]），我们是对$\theta$求梯度，那里假设奖励与$\theta$无关，这里的”奖励“与$\theta$有关，因此出于严谨还是应该另外推一下。

目标是$max_{\pi_\theta} \mathbb{E}_{x\sim D,y\sim\pi_\theta(y|x)}[r_\phi(x, y)-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}]$

先对$\mathbb{E}_{y\sim\pi_\theta(y|x)}[r_\phi(x, y)]$求梯度，由之前策略梯度的推导，可以得到$\nabla_\theta \mathbb{E}_{y\sim\pi_\theta(y|x)}[r_\phi(x, y)]=\mathbb{E}_{y\sim\pi_\theta(y|x)}[ r_\phi(x, y)\nabla_\theta \log\pi_\theta(y|x)]$

再对$\mathbb{E}_{y\sim\pi_\theta(y|x)}[-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}]$求梯度，得到
$\nabla_\theta \mathbb{E}_{y\sim\pi_\theta(y|x)}[-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}]=\mathbb{E}_{y\sim\pi_\theta(y|x)}[-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}\nabla_\theta \log\pi_\theta(y|x)]+\mathbb{E}_{y\sim\pi_\theta(y|x)}[-\beta\nabla_\theta \log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}]$
其中第二项$-\beta\nabla_\theta \log\pi_\text{ref}(y|x)$和theta无关，是0. 而$\mathbb{E}_{y\sim\pi_\theta(y|x)}[-\beta\nabla_\theta \log{\pi_\theta(y|x)}]$由[[强化学习笔记#Expected Grad-Log-Prob Lemma]]也是0. 

所以$\nabla_\theta \mathbb{E}_{y\sim\pi_\theta(y|x)}[r_\phi(x, y)-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}]=\mathbb{E}_{y\sim\pi_\theta(y|x)}\left[\left(r_\phi(x, y)-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}\right) \nabla_\theta \log{\pi_\theta(y|x)} \right]$

推出来结果是一样的，总觉得是不是openai他们一开始就有意设计成这样的？
#### 策略梯度和ppo推导
上面是把生成的回答序列y视作一个动作，这在推导上是没问题的。因为这种单轮的rlhf本来就不涉及与外界的交互，因此无论把一个序列视作动作，还是一个token视作动作，都是可以的。
但是生成token是序列决策问题，它还涉及一个credit assignment的问题，就是搞清楚各个token对最终奖励的贡献有多大，对贡献大的token加大力度奖励。这就是要去做好每个token的优势函数的估计。
$\mathbb{E}_{y\sim\pi_\theta(y|x)}\left[\left(r_\phi(x, y)-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}\right) \nabla_\theta \log{\pi_\theta(y|x)} \right]=\mathbb{E}_{y\sim\pi_\theta(y|x)}\left[\left(r_\phi(x, y)-\beta\log{\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}}\right) \sum_i \nabla_\theta \log{\pi_\theta(y_i|\cdot)} \right]$
这里把$\nabla_\theta \log{\pi_\theta(y|x)}$拆成token，$y_i$表示各个生成的回答token，$\cdot$表示$y_i$对应的所有上文。和强化学习里对轨迹推策略梯度的情况一样，这里的每个token拿到的权重都是相同的，都是整个轨迹的奖励。因此接下来就要考虑奖励的位置，改写成reward-to-go形式，更进一步训练一个value model来用GAE估计优势函数。

如何估计优势函数我们先跳过，假如优势函数是$A_i$，那最后估计的梯度就是

$\mathbb{E}_{y\sim\pi_\theta(y|x)}\left[\sum_i A_i \nabla_\theta \log{\pi_\theta(y_i|\cdot)} \right]$

这里估计的梯度，每更新一次，都要从最新策略里重新采样。为了能在相同的数据上多更新几次，就需要ppo-clip啦。
梯度可以被近似成下面这样，$\pi_s$表示采样回答的旧策略。
$\mathbb{E}_{y\sim\pi_s(y|x)}\left[\sum_i A_i \nabla_\theta \pi_\theta(y_i|\cdot) / \pi_s(y_i|\cdot) \right]$
对应的代理目标函数就是
$\mathbb{E}_{y\sim\pi_s(y|x)}\left[\sum_i A_i  \pi_\theta(y_i|\cdot) / \pi_s(y_i|\cdot) \right]$
然后使用clip来约束$\pi_\theta$不要离$\pi_s$太远。具体方法是用clip对梯度做截断。
$E_{y\sim\pi_s(y|x)} \left[ \sum_i \min \left( \frac{\pi_\theta(y_i|\cdot)}{\pi_s(y_i|\cdot)} A_i, \text{clip} \left( \frac{\pi_\theta(y_i|\cdot)}{\pi_s(y_i|\cdot)}, 1-\epsilon, 1+\epsilon \right) A_i \right) \right]$
然后对$\theta$求梯度就好了。具体在代码里就是对损失(梯度)加权，然后反向传播。

#### 优势函数估计

这里其实是最微妙的地方。可以有很多种用奖励估计优势函数的方法。查阅了openrlhf里ppo的代码，总结如下。

1. kl散度带来的“奖励”被视为对token的奖励，和结尾对整个句子的奖励合并在一起。使用GAE估计优势函数。而用GAE估计优势函数意味着
	1. 当前token往后所看到的奖励都是打折扣的
	2. 需要训练critic网络来估计价值。critic网络的目标则是使用类似GAE的方法估计出来当前状态的价值，然后以此为目标训练
2. 



有几个问题：
1. kl散度带来的”奖励“和整个句子末尾的奖励合在一起合理吗？之后token的kl散度奖励可以忽略掉吗？效果如何？
2. 







而这种情况下的最优策略不需要强化学习，可以解析的通过奖励函数和KL散度约束表示出来。反过来，就可以用当前策略表示出奖励函数，然后代入到优化奖励函数的式子里，从而直接优化当前策略。
就连openAI自己也知道这个最优策略可以写出来。它们的[\[1909.08593\] Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)中有这样一段

![[Pasted image 20231219130015.png]]
![[Pasted image 20231219130039.png]]
最优策略$\pi_\text{opt}$正比于KL散度中的参考策略$\rho$(就是没调的语言模型)采样出来的结果，然后按照奖励函数的系数按照概率做个拒绝采样。然后他们比了比生成的样本在奖励模型上获得的平均奖励，发现强化学习出来的结果都不如最优策略。
![[Pasted image 20231219130205.png]]
既然通过奖励模型拒绝采样出来的结果就是对应的最优策略$\pi_\text{opt}$采样出来的结果，那么直接用这些拒绝采样的结果来优化原来的语言模型不是也可以吗？









## 如何实现




## 训练技巧


大佬的实践经验总结：
[如何正确复现 Instruct GPT / RLHF?](https://zhuanlan.zhihu.com/p/622134699)
[Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://difficult-link-dd7.notion.site/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f)
自己总结一下：
- 把kl散度并入奖励
	- 在算每个动作的kl散度奖励时，用[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) 提到的无偏低方差kl散度估计器估计。（草....这合理嘛。这其实是启发式的把上文推导中每个token的$\log{\frac{\pi_\theta(y_i|x,y_{<i})}{\pi_\text{ref}(y_i|x,y_{<i})}}$ 视作一个单样本的对$\mathbb{D}_{KL}[\pi_\theta(y_i|x,y_{<i})||\pi_\text{ref}(y_i|x,y_{<i})]$的估计，因此可以被替换成另一个估计器）[code](https://github.com/OpenRLHF/OpenRLHF/blob/07c34b303e86fcdffa755e9e25c28cca5b44c675/openrlhf/models/utils.py#L29)
- 用GAE来估计优势函数，实践中设置$\lambda=1$，也就是折扣为$\gamma$的 reward-to-go - V(t). V(t)不一定准，作为baseline减去
- 添加sft loss。一种说法是避免response distribution 偏移reward model训练数据集的分布太多。另一种说法是保留rl之前的能力
- 用sft后的模型初始化policy model，用reward model初始化critic model
- Adam learning rate #todo 
- mini batch update
- value function clip，用clip限制当前价值函数的活动范围不要离旧价值函数太远。用mse拟合当前估计出来的价值
- reward normalization and cliping 
- advantage normalization in a batch
- small training epoch，强化学习时只在采样出来的回答上过一遍，即epoch=1. 样本效率真低。据说训练时80%的时间都用于生成回答
- 在一开始训练时冻结actor model，很合理的技巧，因为毕竟要初始化critic model. 不过critic model作为baseline真的用处那么大嘛？
- reward baseline，一种对奖励归一化的方法。是说奖励r(x,y)高，可能是因为问题x的所有回答奖励都偏高，那么有必要对任务x进行归一化。具体说就是减去x对应的平均奖励。应该可以提前算出来。
- 先拒绝采样finetune，然后ppo



[RLHF 及其变体 Iterative DPO/RLOO/GRPO/REINFORCE 算法和工程分析](https://zhuanlan.zhihu.com/p/714364995)


GRPO
[\[2402.03300\] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

## rl with ai feedback
[Constitutional AI: Harmlessness from AI Feedback \\ Anthropic](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)

[Specific versus General Principles for Constitutional AI \\ Anthropic](https://www.anthropic.com/research/specific-versus-general-principles-for-constitutional-ai)

[\[2307.12950\] RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment](https://arxiv.org/abs/2307.12950)

## off-policy的正确回答

在看Qwen2.5 math的技术报告时，发现有些题目模型就是做不对，因此强化学习时全是负例。面对这种情况一种方法是 可以过滤掉不会的题目，先用容易的题目训练，然后慢慢加大难度。

仔细想了想，多臂赌博机也存在这种问题。比如说有个动作概率很小，但是奖励很大。那么比起让它大量采样后偶然做出这个动作发现这个奖励，不如直接告诉它做这个动作能有很高的奖励。估计梯度时直接在训练样本里加入这个动作，再乘上当前动作对应概率做拒绝采样即可，即$\frac{P_\theta(x)}{1}\nabla_\theta \log P_\theta(x) R(x)$. 这里假设动作x是从一个总是做动作x的分布里采样出来的。

因此对语言模型来说，我们可以通过：
- 给问题x，输出解答y。y错了，于是告诉它错在哪让它改正，然后输出正确答案y'. 使用(x,y)和(x,y')一起加入强化学习训练。也就是说不是让它自己采样发现正确回答，而是加入某种提示来得到正确回答。
这里的问题是y'是用$\pi_{other}$生成的，是off-policy的，直接加进去梯度估计不准了。于是同样考虑是否能用拒绝采样，乘一个系数。

$\mathbb{E}_{y\sim\pi_\theta(y|x)}[ r_\phi(x, y)\nabla_\theta \log\pi_\theta(y|x)]+\mathbb{E}_{y\sim\pi_{other}(y|x)}[\frac{\pi_\theta(y|x)}{\pi_{other}(y|x)} r_\phi(x, y)\nabla_\theta \log\pi_\theta(y|x)]$
在实际训练时，外面的期望都是通过采样获得的。这里可能会有的问题是$\frac{\pi_\theta(y|x)}{\pi_{other}(y|x)}$可能相当小，算这玩意的梯度啥用没有。这方面可能需要测一下，或者在prompt里加入“请对答案进行尽可能少的修改”


如果忽略这一点，就可以假装y'是参考模型$\pi_s$说的，直接把ppo依样套上去了。

$\frac{\pi_s(y|x)}{\pi_{other}(y|x)} \min \left( \frac{\pi_\theta(y_i|\cdot)}{\pi_s(y_i|\cdot)} A_i, \text{clip} \left( \frac{\pi_\theta(y_i|\cdot)}{\pi_s(y_i|\cdot)}, 1-\epsilon, 1+\epsilon \right) A_i \right)$
$\pi_s$指的是参考模型，也就是实际采样，约束$\pi_\theta$的策略。另一个就是不能拿y'训练价值函数（优从而估计GAE）。优势函数A在ppo的推导中是与$\pi_s$相对应的。之所以在ppo里需要约束参考策略和当前策略，就是为了确保让参考策略对应的优势函数$A_s$能近似$A_\theta$

然而语言模型训练时一般 A=reward-to-go - baseline. 而reward-to-go只和轨迹有关。这样的话，不管是什么策略展开出来的轨迹算出来的优势函数，区别都只有baseline（价值函数）....
嘛，总之就是对这种off-policy的回答，A采用和正常轨迹同样的方法去计算。只需要在clip项外面乘一个系数就行了。

不知道自己有没有表述清楚，自己的理解是不是有问题。也不知道是否实用，先记录在这里。
果然还是应该多一点实践的经验才行。

2024/11/14更新：翻johnschulman2的twitter时看到这篇文章。里面采用了和上面一样的分解思路，并据此提供了一种在ppo更新策略时提升batch size的方法(epoch=1)。
[\[2110.00641\] Batch size-invariance for policy optimization](https://arxiv.org/abs/2110.00641)


## 与GAN的相似性

大语言模型是生成器，它的生成结果为一串token。奖励模型是判别器，它给生成结果打分。
用人类偏好训练奖励模型，其实就是训练一个更好的判别器。自己对GAN不是特别熟悉，下面按自己的理解简单做一下对比。



训练图像生成器时，通过采样已知分布的隐藏变量x，生成图像。然后通过奖励模型反向传播，更新生成器的参数以最大化判别器的分数。






那么能不能使用类似GAN的方法训练语言模型呢？

语言模型之所以没有用类似GAN的方式，直接从RM的反向传播中获得梯度的原因可能有以下几点：
1. 语言模型的“采样”发生在输出token。这个采样过程是不可微的（大概）。（这里没细想，说不定可以通过将采样过程重参数化或者STE等方式让它可微？）而图像GAN的采样是在神经网络的输入，采样一个已知分布x获得的。
2. GAN只适用于奖励模型可微的情况（不过在rlhf这里不是什么大问题）
3. 语言模型的采样比较贵，耗时间。为了重复使用之前的采样结果多更新几步，可能还是得ppo（可能？）

如果用了类似GAN的方式去做，可能的好处有：
1. 从RM的反向传播中，我们或许可能可以更好的对token做“credit assignment”，即知道哪个token对最终奖励的影响最大。这样就不需要整一个critical model来做信用分配了。（也许不太实用，但或许可以做一些对奖励模型的可解释性工作？）
2. 



最后简单搜索了一下相关论文，放在下面：

[APO｜利用_GAN_的思想训练_RLHF_中的RM](https://zhuanlan.zhihu.com/p/674776494)
看评论区对喷好有意思。
这是使用gpt-4的回复作为金标准，训练判别器区分gpt-4的回复和自己的回复。
然后调整llm生成器的方法就可以使用ppo,dpo等方法。
但我上面的想法是用类似GAN的方式调整llm生成器，所以还是有些不同。