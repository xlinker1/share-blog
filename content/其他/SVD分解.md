---
draft: false
tags:
---

[PCA(主成分分析) 和 SVD (奇异值分解) - 知乎](https://zhuanlan.zhihu.com/p/92507652)

先说主成分分析PCA：

假设我有n个数据点，每个点m维。每列代表一个点，组成一个矩阵X (mxn)
对行取平均，相当于计算数据点的质心。然后对所有数据点减去这个质心，让它们均值为0. 得到矩阵Y (mxn)
协方差矩阵 $C=\frac{1}{n}YY^T$ ，维度为mxm。对角线上描述的是数据点各个维度自身的离散程度，也就是方差，非对角线元素是协方差。两个维度之间的协方差可以除以各自维度的标准差来进行归一化，归一化后的结果是相关系数，在-1~1之间。为1是正相关，为-1是负相关，为0是没有明显的线性关系。
协方差矩阵C描述了数据点在空间各个方向上分布的离散情况。那么如何通过它具体求得空间上在单位向量v方向上的数据离散程度呢？我们知道$v^TY$表示每个数据点在v向量方向上的投影，那么 $v^TY(v^TY)^T/n$ 就是该方向上的方差，也就是离散程度。用C矩阵来求就是$v^TCv$


C是正交矩阵，总可以求出它的特征值和特征向量。即可以写成$C=V\text{特征值}V^T$
其中，V是单位正交矩阵。特征值矩阵是对角阵，其中每一个元素对应V中那一列特征向量的特征值。特征值对应该特征向量方向上的方差。
另一方面，可以将V理解成一个旋转矩阵。通过对所有Y左乘$V^T$进行旋转，可以让数据点各个维度线性不相关，各个维度各自的方差对应C的各个特征值。


除了通过$V^TY$来进行变换，我们也可以只投影部分方向，即$V[:,:3]^TY$. 这里只选取了前三列特征向量进行投影。假如这三列特征向量对应的特征值最大，那么就意味着投影完的数据点方差最大，也就最大程度的保留了信息。这些就是数据点的主成分
继续左乘对应的特征向量就可以恢复原来的数据点，$V[:,:3]V[:,:3]^TY$.
也就是说，直接左乘$V[:,:3]V[:,:3]^T$矩阵可以将所有数据点投影到这三列特征向量张成的子空间中。


由此可见，V是数据点Y中非常有用的一部分。
SVD分解告诉我们，任何矩阵A(mxn)都可以被分解成三个部分$A=U\Sigma V^{T}$. 其中U(mxm),V(nxn)是单位正交矩阵。$\Sigma$(mxn)是对角矩阵，因为不为方阵，所以可能存在全为0的行和列。假如m>n，存在全为0的行，就可以把U中的后几列删去，从而U(mxn),$\Sigma$(nxn),V(nxn)
它和A乘A的转置的特征分解有对应关系如下
$AA^T=U\Sigma\Sigma^TU^T$
$A^TA=V\Sigma^T\Sigma V^T$
对应到我们的数据矩阵Y，也可以通过SVD分解分解成这种形式。即$Y=V\Sigma \text{乘另一个特征矩阵}$


