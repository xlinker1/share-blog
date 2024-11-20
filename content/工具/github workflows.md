---
draft: false
tags:
---

## 基础用法

先创建一个git仓库，在仓库的根目录创建下面这个文件，然后push到github上。
点击仓库页面的actions标签，看看效果。

.github/workflows/meo.yml
```yml
on: push

jobs:
  job1:
    runs-on: ubuntu-latest
    steps:
      - run: pwd
      - run: ls
      - run: node --version
      - run: npm --version
      - run: tsc --version
      - run: npm i -g typescript
      - run: docker --version
      - run: docker image ls
      - run: docker pull nginx
      - run: docker image ls

  job2:
    runs-on: windows-latest
    steps:
      - run: pwd
      - run: ls
      - run: node --version
      - run: npm --version
      - run: npm i -g typescript
      - run: tsc --version
      - run: docker --version
```

上面yml文件的意思是在push时触发工作流，工作流里有两个互相不依赖的job，分别叫job1和job2，在不同的环境里运行。环境镜像的详细信息和自带软件包可以在workflow的输出里查看。
![[Pasted image 20240728115218.png]]

steps描述了顺序执行的命令，上面打印了两个镜像自带软件的版本，安装了tsc，还拉取了docker镜像。

印象里github workflow命令运行在docker容器里，所以看到image里自带docker后有点惊了，难道docker里还能套docker？但是仔细看了看，github action用的是虚拟机镜像。镜像里不仅有docker，还有各种编程语言的包管理器和工具，甚至还有浏览器

于是想到是不是可以用workflow在github的虚拟机环境里拉取并导出镜像，然后把导出的镜像作为build的产物下载下来。但是查了一下发现免费账户单次下次build产物不能大于500MB，每个月使用虚拟机时长不能大于2000s，哎，不能白嫖了。
[[各种网络问题#docker代理]]


## 常见应用

#### 把静态网页部署到github page上

可以看看这个demo项目的workflow文件的修改历史。
[History for .github/workflows/meo.yml - xlinker1/typescript\_demo · GitHub](https://github.com/xlinker1/typescript_demo/commits/master/.github/workflows/meo.yml)

虽然workflow看上去很复杂，但是秉承一个没用到就不学的精神，先把用到的记录如下做个备忘。

- jobs里的每一个job并行执行。可以在job1里添加needs: job2 来添加依赖关系
- job里的steps包含一个列表，列表里面是字典。字典里面描述动作。执行时按照列表顺序执行动作。
	- 动作可以是`run:`后面跟单行/多行命令。可以在字典里通过`env:`，`shell:`来设置命令的环境变量和shell，也可以通过`if:`来条件判断。
	- 动作也可以是`uses: actions/checkout@v4`，执行其它仓库中的action.yml。类似于一个函数。可通过`with:输入参数键值对`来输入参数。
		- action.yml和workflows里的yml写法不同。有inputs:和outputs: ，类似于函数。


#### 自动打包docker镜像


