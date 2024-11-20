---
draft: false
tags:
---

## attention_mask真有用吗？

transformer在算自注意力的时候，大致上是$softmax(QK^T)V$
这里假设Q矩阵的每一行为一个token embedding，从上到下为token 0, token 1, token 2等等。为了确保token 1只能看到自己和之前的token，看不到之后的token2，我们会把$QK^T$矩阵的上三角设置成-inf，这样在softmax之后，这些位置的注意力就被忽略了。这就是casual mask。

在使用transformers库的时候，发现模型总是要输入attention_mask。输入数据长度不同，为了补齐短的序列，会在它后面填充特殊token`<pad>`。attention_mask在tokenizer额外填充的pad token处为零，其余为1. 
想要让pad不影响训练，主要是两个方面：1. 不要参与前向计算，不要影响正常token的输出 2. 不要参与损失计算
第一点，由于pad总是在序列后面填充，所以肯定不会影响前面token的输出。
第二点，想要pad token不要参与损失计算，最简单的办法就是在输出标签为pad token时把损失设置为0. 可以在pytorch的cross entropy里设置特殊的ignore token，也可以在损失算完后乘个0. 
然而这玩意叫attention_mask，顾名思义，理应是在前向计算注意力时和casual mask一起添加的。试着查看了库里gptj的代码，发现也确实是这样。它是在mask $QK^T$的列，把输入为`pad`的列给mask掉了。但这有什么意义呢？如果只是忽略`pad`输入，但是在输出参与损失计算还是会有影响啊。如果忽略了输出为pad的损失计算，pad又只是加在序列末尾，那么加不加attention_mask有意义吗？

当然，如果pad token加在句子中间，可能就有必要用attention_mask了。

不知道是否理解有问题，还是新手，希望大伙不吝赐教一下~

