# 基本性能

基本分为两种数据集合，第一种是二值图片数据；第二种是自然图片数据。

[Default Setting](http://mlserver.ipwx.me:7897/5de601d3ceacfbad434dc02a/)

## NLL

|      | CIFAR10 | CELEBA(64x64) | MNIST(Bernouli) | FASHION | LSUN |
| ---- | :-----: | ------------- | --------------- | ------- | ---- |
| 1    |         |               |                 |         |      |
| 2    |         |               |                 |         |      |
| 3    |         |               |                 |         |      |
| 4    |         |               |                 |         |      |

s

## nll and FID, IS when z_dim is changed on cifar10

| Z_dim         | Nll  | FID  | IS   |
| ------------- | ---- | ---- | ---- |
| 64            |      |      |      |
| 128           |      |      |      |
| 256           |      |      |      |
| 512           |      |      |      |
| 1024(default) |      |      |      |

# 细节指标

## Z计算的误差 和 NLL计算的误差

重复计算Z和NLL，统计方差，至少10次，每次1M的采样数：

Cifar another Z:[418.03 $\pm$ 4.98](http://mlserver.ipwx.me:7897/5df8d816ceacfbad434dc062/)

Cifar Z: [-489.49$\pm$ 0.00](http://mlserver.ipwx.me:7897/5e01fe96f9a08f4f544dc14f)

Cifar NLL:[6370.46$\pm$ 4.82](http://mlserver.ipwx.me:7897/5df8d816ceacfbad434dc062/)

Mnist Z:[]

Mnist NLL:[]



## ELBO，Reconstruction，KL(q(z|x)||p(z))

这个指标是ELBO和Reconstruction的差值，也是KL(Q||P)的上界：

Cifar：



CelebA：



Mnist：



Fashion：



##  训练时Loss的曲线，用小样本进行的估计

由于我们的算法没有使用验证集，只需要列出训练时的loss的变化就行了，如果觉得一根曲线太少，可以附加上ELBO，判别器的变化等曲线（在训练时已经导出了）





## 使用WGAN训练出的Discriminator直接进行VAE训练的NLL及其Loss曲线

dmm_cifar10_nD.py







# Out of distribution

我们在Cifar10和Mnist上进行模型的训练，那么期望在Cifar的训练集数据上的NLL<Cifar的测试集数据上的NLL<<在SVHN数据集上的NLL。如果有反常的现象，那么我们就认为是出现了out-of-distribution的现象。这一检测必须要在自然图片数据集和二值图片数据集上都进行，即Cifar10训练，SVHN测试；Mnist训练，NotMnist测试。

|        | cifar10 vs svhn                                              | cifar10 vs imagenet                                          | cifar10 vs lsun                                              |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| nll    | [auc: 0.04, ap: 0.16](http://mlserver.ipwx.me:7897/5df254bff9a08f4f544dc0ec) | [auc: 0.86, ap: 0.48](http://mlserver.ipwx.me:7897/5e143f37f9a08f4f544dc1dd) | [auc: 0.60, ap 0.60](http://mlserver.ipwx.me:7897/5e1197989eb234c36e4dc0a1) |
|        | auc: 0.92, ap: 0.80                                          | auc: 0.81, ap: 0.39                                          | auc: 0.53, ap: 0.52                                          |
| energy | auc: 0.70, ap: 0.39                                          | auc: 0.36, ap: 0.12                                          | auc: 0.47, ap: 0.50                                          |
| norm   | auc: 0.86, ap: 0.71                                          | auc: 0.30, ap: 0.11                                          | auc: 0.45, ap: 0.48                                          |
| all    | auc: 0.93, ap: 0.81                                          | auc: 0.64, ap: 0.30                                          | auc: 0.49, ap: 0.51                                          |

Cifar和SVHN的NLL和对应的分布图：

WGAN（energy, grad_norm）：

​	WGAN-div, WGAN-gp, WGAN-both:

​	

dmm_nD（nll, energy, grad_norm）：

​	dmm_nD-div, dmm_nD-gp, dmm_nD-both:



dmm（nll, energy, grad_norm）：

​	dmm-div, dmm-gp, dmm-both:



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



