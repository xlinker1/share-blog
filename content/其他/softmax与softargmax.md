---
draft: false
tags:
---


注意看下面这个代码示例，通常我们说的softmax应该叫softargmax，而$\log\sum\exp$才是softmax：

```python
import numpy as np

def soft_max(arr):
    return np.log(np.sum(np.exp(arr)))

def soft_argmax(arr):
    exp_arr = np.exp(arr)
    max_index = np.random.choice(len(arr), p=exp_arr/np.sum(exp_arr))
    # 从exp_arr/np.sum(exp_arr)分布中采样
    return max_index
    
arr=np.array([1,2,3,4,3])
print('soft_max:', soft_max(arr), 'max:', np.max(arr))
print('soft_argmax:', soft_argmax(arr), 'argmax:', np.argmax(arr))

# soft arg max 给出选择最大下标的概率
l = np.zeros_like(arr )
for _ in range(1000):
    l[soft_argmax(arr)]+=1
print(l)
```

[Softmax function - Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)