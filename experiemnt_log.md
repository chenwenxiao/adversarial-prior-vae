# 基本性能



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

重复计算Z，统计方差：

Cifar:[]

Mnist:[]



## KL(q(z|x)||p(z))

这个指标是ELBO和Reconstruction的差值，也是KL(Q||P)的上界：

Cifar：

CelebA：

Mnist：

Fashion：



# Out of distribution

Cifar和SVHN的NLL和对应的分布图：



Mnist和NotMnist的NLL和对应的分布图：



# 图

## 采样图

Cifar：



CelebA：



Mnist：



Fashion：



## 重构图

Cifar：



CelebA：



Mnist：



Fashion：



## 塔模型图

Cifar：



CelebA：



Mnist：



Fashion：



