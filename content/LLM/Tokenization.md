---
draft: false
tags:
---

Tokenization的目的是把文本转化成一个个Token，或者说词。它是和语言模型训练相互分离的一个过程，语言模型训练时，一般只是通过下标获取对应token的embedding。
它一般有几个指标：
1. 给定词表预算的情况下，压缩率高。
	- 压缩率高，相同长度的文本在Transformer眼里就越短。因此也就能在固定上下文支持的token数的情况下支持更长的文本。推理因为是自回归的，推理相同长度文本也只用更少次数的前向推理，因此也更快。
2. 无损，就是希望文本经过编码之后能完全恢复
3. 训练速度
4. 编码速度和解码速度

自己在对Tokenizer有一点初步了解的时候就想，既然语言模型的训练等于压缩（预测的越准确压缩率越高），那么是否有可能在字节上用Transformer训练一个语言模型，然后基于这个语言模型对各个字节出现概率的估计，造一个Tokenizer呢？

## BPE

[Let's build the GPT Tokenizer - YouTube](https://www.youtube.com/watch?v=zduSFxRajkE)
tokenizer介绍与BPE算法解读



## 语言模型+维特比解码

[【中文分词系列】 5. 基于语言模型的无监督分词 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/3956)
[BytePiece：更纯粹、更高压缩率的Tokenizer - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/9752)

[GitHub - bojone/bytepiece: 更纯粹、更高压缩率的Tokenizer](https://github.com/bojone/bytepiece/tree/main)

仔细思考后，发现这确实是很漂亮的算法。想不出什么改进的办法
1. 语言模型完全从数据中训练得出
2. 基于语言模型对每个字节进行标注，标注当前字节是词的第n个字节的概率
3. 用[[维特比算法]]来获取全局最高概率的词性标注

文中训练的语言模型是用n-gram，通过在训练文本上统计获得。作者推荐是使用6-gram。大于6-gram的条件概率使用6-gram做近似。

这样的话当然就可以在字节上训练一个transformer作为语言模型。我觉得行，但似乎不太有必要....


