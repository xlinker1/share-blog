---
draft: false
tags:
  - todo
---
[pytest文档](https://docs.pytest.org)
## 使用pytest

基本使用
```
python3 -m pytest --pdb -k "nn_epoch_ndl"
```

命令行参数：
-k "nn_epoch_ndl"  -k 后面跟筛选字符串，指定运行哪个测试。筛选目标包括函数名和文件名
--collect-only 显示选中的测试样例
-s不隐藏代码里的print输出
-l打印局部变量
--pdb 非常有用，在遇到错误时进入pdb环境来找bug
--trace 在test开始时就进入pdb
--tb=style  如何显示失败的测试用例的回溯信息。style可以是no,line,short,long,native(不显示pytest的额外信息)
-v 更详细的显示测试用例的信息，-q相反


筛选字符串可以是简单的函数名里的字符串，也可以是这样嵌套的包含and or not等逻辑运算符的语句
"(permute or reshape or broadcast or getitem) and cpu and not compact"


(做lab的时候感慨，还是多写单元测试，多调试一下比较好，不然自己实现的对不对真的心里没底)



