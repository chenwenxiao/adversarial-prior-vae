import code.experiment.utils as utils
from code.experiment.datasets.omniglot import load_omniglot
from code.experiment.datasets.celeba import load_celeba
import tfsnippet as spt
import numpy as np
import matplotlib.pyplot as plt

def compute(img_path,dataset_name):
    dataset={'mnist':spt.datasets.load_mnist,
             'cifar10':spt.datasets.load_cifar10,
             'fashion':spt.datasets.load_fashion_mnist,
             'omniglot':load_omniglot,
             }
    gen_img = np.load(img_path)
    for k,v in gen_img.items():
        print(k)
    gen_images = gen_img['mala_img']
    if dataset_name=='celeba':
        train_x,test_x,validation_x=load_celeba()
        real_images = np.concatenate((train_x,test_x,validation_x),axis=0)
    else:
        (train_x,train_y),(test_x,test_y) = dataset[dataset_name]()
        real_images = np.concatenate((train_x,test_x),axis=0)
    limit=min(len(gen_images),len(train_x))
    limit=min(50000,limit)
    limit=100
    base=43304
    real_images=real_images[base:base+limit]
    gen_images=gen_images[base:base+limit]
    print(gen_images[12])
    print(real_images.shape,gen_images.shape)
    result = utils.precision_recall(real_images,gen_images,limit,batch_size=20)
    print(result)
    print('1 precision ',result[0],' recall ',result[1])
    print('2 precision ',result[2],'\nrecall ',result[3])
    utils.plot([(result[2],result[3])],name=dataset_name)
    # for i in result[2]:
    #     print(i,end='')
    # for i in result[3]:
    #     print(i,end='')

if __name__ == '__main__':
    mnist_test='/mnt/mfs/mlstorage-experiments/cwx17/ae/1c/d4747dc47d24cf35d1e5/sample_store.npz'
    cifar_test='/mnt/mfs/mlstorage-experiments/cwx17/f5/1c/d445f4f80a9f68b140e5/sample_store.npz'
    print('mnist')
    compute(mnist_test,'mnist')
    print('cifar10')
    compute(cifar_test,'cifar10')