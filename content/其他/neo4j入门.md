---
draft: false
tags:
---





[Neo4j的安装与使用 | 帅大叔的博客](https://rstyro.github.io/blog/2022/10/11/Neo4j%E7%9A%84%E5%AE%89%E8%A3%85%E4%B8%8E%E4%BD%BF%E7%94%A8/)

### 安装

建议使用docker安装。[Docker - Operations Manual](https://neo4j.com/docs/operations-manual/current/docker/)

自己先是在已有的linux容器里用压缩包安装。[Linux executable (.tar) - Operations Manual](https://neo4j.com/docs/operations-manual/current/installation/linux/tarball/)也不是很复杂，前几次能正常运行，但是忽然就报错，说”inotify不够了“。重启该容器后可以正常运行，但用久了还是会报同样的错，因此还要重启。这样就太麻烦了，因此不如直接用docker安装。

而且社区版只能使用默认的名为neo4j的数据库，不能添加多个图数据库。用docker的话，想要使用多个数据库，只需要启动多个容器就可以了。


### 基本查询语法

[Cypher Cheat Sheet - Neo4j Community Edition](https://neo4j.com/docs/cypher-cheat-sheet/5/neo4j-community) 强烈推荐先看这个，这些语法设计可能是最有趣的了。
[Neo4j的安装与使用 | 帅大叔的博客](https://rstyro.github.io/blog/2022/10/11/Neo4j%E7%9A%84%E5%AE%89%E8%A3%85%E4%B8%8E%E4%BD%BF%E7%94%A8/)



图数据库中就是存节点和边，节点有属性（类似于字典的键值对）和标签，边也有属性和标签。边都是有向边。

MATCH 用于匹配模式。这里的模式可以是节点模式，边模式，或者是节点-边->节点<-边-节点的模式。模式中可以给节点/边/路径命名，这样方便后面引用匹配到的结果。
一个MATCH语句会匹配出很多结果，每个结果就像是一行数据项。

return 用于将MATCH匹配到的结果，按照需要的内容逐行输出。可以全部输出，也可以只输出某个节点，也可以只输出某个节点的属性。

where 用于将MATCH匹配到的结果进行逐行过滤。保留满足条件的行，和关系数据库很像。它还可以作为匹配条件放在match里面

with 和 return很像。return直接将匹配结果以表的形式返回，with则将匹配结果命名为变量，然后输入给下一个匹配。return表示查询的结束，with则放在查询之间（或者开头）

unwind 将一个列表转换成查询中可以一行一行处理的表。






数据导入









### 数据分析和可视化插件

Neo4j的图算法库，如PageRank、Community Detection等，对用户进行聚类分析

可视化工具，如Neo4j Browser和Neo4j Bloom
NEuler [Graph Data Science Playground](https://neuler.graphapp.io)




