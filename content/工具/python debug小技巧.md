---
draft: false
tags:
---
最近在做[这个](https://dlsyscourse.org/assignments/)lab，学会了一些调试小技巧。
## python debug

### 命令行调试
使用pdb
import pdb; pdb.set_trace()
使用关键字breakpoint()，效果和上面那行一样。运行到这行就会自动在pdb环境下调试。
也可以用ipdb，安装后设置环境变量export PYTHONBREAKPOINT=ipdb.set_trace 在代码里运行到breakpoint()时就调用ipdb进行调试。
小技巧：可以利用额外的条件语句在特定的时候进去调试状态。

运行脚本，遇到错误时进入调试模式：
```bash
python -m pdb -c continue myscript.py
# or
python -m pdb -c continue -m myscript
# https://stackoverflow.com/a/2438834
```
或者也可以简单的用python -i myscript.py，停下来后就地进入交互模式。
在python的单元测试框架pytest里也提供这样的功能，如
```
python3 -m pytest --pdb -s -l -k "nn_epoch_ndl"
```
--pdb可以在报错时进入调试，-s不隐藏代码里的print输出，-l打印局部变量，-k指定运行哪个测试。
(做lab的时候感慨，还是多写单元测试，多调试一下比较好，不然自己实现的对不对真的心里没底)

#### pdb调试命令
s(step into the function) n(next line)
l, 打印附近代码
r(return from the function),c(continue)
unt(until) 向前运行到某行
b(breakpoint) cl(clear) 清除断点
tbreak 临时断点

w(where)，显示调用栈
u(up)，d(down)在调用栈上下移动，方便查看各个调用栈的局部变量

j(jump) 跳到某处执行

p(print) ipdb里还有pp(pretty print)

直接输入Enter，会执行上一条命令。输入PDB不认识的命令，PDB会把他当做Python语句在当前环境下执行。
输入interact启动一个python的交互式解释器，使用当前代码的全局命名空间（使用ctrl+d返回pdb）

### 结合vscode调试
[vscode python设置debug ? - 知乎](https://www.zhihu.com/question/35022733/answer/3178874019)

## 在python中对编译好的c++模块进行debug
用debug模式编译c++模块
使用gdb运行python
然后run "test_script.py"
然后就可以设置断点等等
一个例子：

``` bash
gdb python
(gdb) run /path/to/script.py
## wait for segfault ##
(gdb) backtrace
## stack trace of the c code
```
