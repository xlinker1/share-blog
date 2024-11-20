---
draft: false
tags:
---

[Welcome to Quartz 4](https://quartz.jzhao.xyz/)

感谢作者，文档写的真是不错，搜索功能也又快又方便。
下面大概记录一下用它部署笔记，需要经过哪几个步骤，方便和我一样的小白

步骤：
1. 把Quartz这个项目clone下来，安装环境。把自己的obsidian vault放到content文件夹下，用quartz在本地生成静态页面，查看效果
	1. 生成过程可以通过插件和设置来配置，还可以自己写插件（然而不会typescript）
	2. markdown开头标注了`draft: true`的文件不会被生成静态页面。
2. 在github上新建一个repo，先将本地项目的修改commit，然后再将该commit推送到github。github会根据项目里设置的workflow，在你设定的docker环境下用你repo里的代码运行命令重新编译出静态页面，然后对静态页面进行部署。
	1. 注意上一步在本地生成的静态页面是不会被commit的，实际上commit的是原本quartz项目的代码，和你content文件夹下的所有内容。`draft: true`只会控制静态页面的生成，而不会控制这个markdown文件是否被commit。




比较遗憾的一点是官方repo生成的静态页面暂时不支持评论功能。
静态网页本身不存储用户状态，因此也没办法评论。但是评论功能可以通过github repo里的issue和discussion功能实现。也就是显示评论时，静态网页会通过api获取github repo里的discussion数据，发布评论时，静态网页会让你登录自己的github账号，然后把评论转发到discussion里。
要支持评论功能，只需要在静态网页最下面，添加一段脚本，设置api参数就好了。
[8. 给github博客添加评论功能 — zzq's blog](https://zzqcn.github.io/design/rest/add_comment.html)


之前也搞过github pages，部署的时候有个需要注意的地方，记录一下。（stackoverflow首答，开心！）
[Do I need the pages-build-deployment Github Action, when I have another action for my Github Pages? - Stack Overflow](https://stackoverflow.com/questions/72079903/do-i-need-the-pages-build-deployment-github-action-when-i-have-another-action-f/76317125#76317125)


## 小修改记录

### 修改数学公式渲染引擎

默认使用katex，然后发现渲染出现问题，于是在quartz.config.ts中修改：
Plugin.Latex({ renderEngine: "mathjax" }),


### 用gitignore忽略草稿

上面说了obsidian里的草稿在git commit时，还是会添加到commit里。这也就意味着会被推送到github仓库里被大家看到。草稿被别人看到怪不好意思的，于是写了个脚本来根据`draft: true`标签生成content/.gitignore文件，忽略草稿。

但是实际上也可以把本地构建后的html自动推送到仓库里，只是自己不太知道怎么弄。像这个人[GitHub - aarnphm/sites: generated source of aarnphm\[dot\]xyz](https://github.com/aarnphm/sites)


### 添加换行符

生成网页后，发现效果和obsidian里的预览效果不一致。obsidian里一次换行就是换行，换两次行是分段。但在标准markdown里，换一次行会被忽略，换两次行才是换行。生成的网页也是这么处理的，这就导致很多在obsidian里看着还行的文字因为没有换行被挤成了一坨。

f12查看html，发现换两次行分段是通过`<p></p>`表示，段落里的换行在在行末添加`<br>`即可。

自己因为基本没学过typescript，边看边猜相关代码。代码里还都是async，看起来很不好debug

查看quartz代码，发现解析markdown的关键部分采用的是别人的库。然后作者自己针对obsidian的特性增加了一些处理。
仿照这些插件，在解析后的html树里，我试着对所有`<p></p>`节点里的text做如下处理：
1. 如果是多行text，删除原text节点。依据`\n`把原text节点用切成多个text节点，并在中间插入`<br>`元素
2. 如果是单行text，不做处理

其实本来想着是直接修改text元素的文本，行末插入`<br>`就完事了。但是后来一看这样不行，插入的`<br>`会被转义成大于号...

第二天去看，发现有对应的插件....果然还是先问问别人比较好呀。也是，这个问题应该很明显才对
[Inconsistent Line Break Rendering Between Obsidian and Standard Markdown · Issue #1326 · jackyzha0/quartz · GitHub](https://github.com/jackyzha0/quartz/issues/1326)

### npx quartz build --serve

上面这个命令的功能是构建网页，并开一个网页服务。如果这些markdown文件有修改，就即时的重新生成网页。
但是出现了一些小问题，就是修改了笔记之后，似乎是因为缓存还是什么原因无法生成最新的网页，目录也有问题。
然后尝试删除没有被git跟踪的cache文件，发现`rm quartz\.quartz-cache`后，再 npx quartz build --serve 就没问题了。


[quartz build --serve does not update explorer with folder / new notes · Issue #1077 · jackyzha0/quartz · GitHub](https://github.com/jackyzha0/quartz/issues/1077)


### bug

部署了才知道，bug真不少....不仅评论功能没生效，原本的多行公式渲染也出bug了。早知道不更新了....
如果要获得最好的观看体验，推荐直接把 https://github.com/xlinker1/share-blog/tree/v4/content 下的内容下载下来用obsidian打开....