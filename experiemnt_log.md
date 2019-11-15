# 基本性能

基本分为两种数据集合，第一种是二值图片数据；第二种是自然图片数据。





## NLL

|      | MNIST | FASHION | CIFAR10 | CELEBA |
| ---- | :---: | ------- | ------- | ------ |
| 1    |       |         |         |        |
| 2    |       |         |         |        |
| 3    |       |         |         |        |
| 4    |       |         |         |        |

## FID

Final FID by tower model:





## fid and nll when std is changed

| Log_std |  MALA  |  ORIGIN | GAN  |
| ------- | ---- | ---- | ------ |
| 2.0     |      |      |        |
| 1.5     |      |      |        |
| 1.0     |      |      |        |
| 0.5     |      |      |        |
| 0.0     |      |      |        |
| -1.0    |      |      |        |
| -3.0    |      |      |        |
| -6.0    |      |      |        |
| -10.0   |      |      |        |



# 细节指标

## Z计算的误差

重复计算Z，统计方差，至少10次，每次1M的采样数：

Cifar:[]



Mnist:[]



## KL(q(z|x)||p(z))

这个指标是ELBO和Reconstruction的差值，也是KL(Q||P)的上界：

Cifar：



CelebA：



Mnist：



Fashion：



##  训练时Loss的曲线，用小样本进行的估计

由于我们的算法没有使用验证集，只需要列出训练时的loss的变化就行了，如果觉得一根曲线太少，可以附加上ELBO，判别器的变化等曲线（在训练时已经导出了）





## 





# Out of distribution

我们在Cifar10和Mnist上进行模型的训练，那么期望在Cifar的训练集数据上的NLL<Cifar的测试集数据上的NLL<<在SVHN数据集上的NLL。如果有反常的现象，那么我们就认为是出现了out-of-distribution的现象。这一检测必须要在自然图片数据集和二值图片数据集上都进行，即Cifar10训练，SVHN测试；Mnist训练，NotMnist测试。

Cifar和SVHN的NLL和对应的分布图：



Mnist和NotMnist的NLL和对应的分布图：



# 图

## 采样图

这里的采样图必须是通过由模型直接导出的图片。

Cifar：



CelebA：



Mnist：



Fashion：



## 重构图

除了每个数据集的重构图以外，对于log_std变化的实验也要列出重构图随着log_std变化的情况。比如，std较小的情况下，重构图很清晰；随着std增大，重构图慢慢变得模糊。

Cifar：



CelebA：



Mnist：



Fashion：



## 塔模型图

这里的采样图必须是通过由塔模型直接导出的图片，我们人工筛选一下尽量保留好看的图片。

Cifar：



CelebA：



Mnist：



Fashion：



