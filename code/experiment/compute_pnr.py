import code.experiment.utils as utils
from code.experiment.datasets.omniglot import load_omniglot
from code.experiment.datasets.celeba import load_celeba
import tfsnippet as spt
import numpy as np
import matplotlib.pyplot as plt

def compute(img_path,dataset_name,savename='',savedir=None):
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
    # limit=100
    base=0
    real_images=real_images[base:base+limit]
    gen_images=gen_images[base:base+limit]
    print(gen_images[12])
    print(real_images.shape,gen_images.shape)
    result = utils.precision_recall(real_images,gen_images,limit,batch_size=20)
    print('2 precision ',result[2],'\nrecall ',result[3])
    # utils.plot([(result[2],result[3])],name=f'{savename}-{dataset_name}')
    # utils.plot([(result[0],result[1])],name=f'{savename}-{dataset_name}-2')
    if savedir != None:
        if not os.path.exists(f'./{savedir}'):
            try:
                os.makedirs(f'./{savedir}')
            except Exception:
                print(f'--{savedir} failed to create')
                pass
        np.savez(f'./{savedir}/{savename}-{dataset_name}', precision=result[2], recall=result[3])
    else:
        np.savez(f'./{savename}-{dataset_name}',precision=result[2],recall=result[3])
    return result[2],result[3]
    # np.savez(f'./{savename}-{dataset_name}-2',precision=result[0],recall=result[1])
    # for i in result[2]:
    #     print(i,end ='')
    # for i in result[3]:
    #     print(i,end='')

import os

def plot_in_one(path,picname='total'):
    names = os.listdir(path)
    pr_pairs=[]
    labels=[]
    for name in names:
        if 'npz' in name:
            pr_pair = np.load(os.path.join(path,name))
            pr_pairs.append((pr_pair['precision'],pr_pair['recall']))
            labels.append(name[:-4])
    # plot
    utils.plot(pr_pairs,labels,name=picname)

vaepp_generated_samples_mnist = [
    "/mnt/mfs/mlstorage-experiments/cwx17/ae/1c/d4747dc47d24cf35d1e5/sample_store.npz", # 1.0 FID: 11.98
    "/mnt/mfs/mlstorage-experiments/cwx17/63/2c/d4747dc47d2411e897e5/sample_store.npz", # 0.8 FID: 14.82
    "/mnt/mfs/mlstorage-experiments/cwx17/57/2c/d445f4f80a9f71e897e5/sample_store.npz", # 0.5 FID: 33.61
    "/mnt/mfs/mlstorage-experiments/cwx17/93/2c/d4747dc47d2412cc97e5/sample_store.npz", # 0.2 FID: 103.33
    "/mnt/mfs/mlstorage-experiments/cwx17/73/2c/d4747dc47d2452e897e5/sample_store.npz", # 0.1 FID: 135.11
]

vaepp_generated_samples_cifar10 = [
    "/mnt/mfs/mlstorage-experiments/cwx17/f5/1c/d445f4f80a9f68b140e5/sample_store.npz",  # 1.0 FID: 71.00
    "/mnt/mfs/mlstorage-experiments/cwx17/8f/0c/d4e63c432be9339897e5/sample_store.npz",  # 0.8 FID: 91.01
    "/mnt/mfs/mlstorage-experiments/cwx17/7f/0c/d4e63c432be9d29897e5/sample_store.npz",  # 0.5 FID: 147.94
    "/mnt/mfs/mlstorage-experiments/cwx17/62/1c/d434dabfcaec229897e5/sample_store.npz",  # 0.2 FID: 211.16
    "/mnt/mfs/mlstorage-experiments/cwx17/53/2c/d4747dc47d24529897e5/sample_store.npz",  # 0.1 FID: 267.80
]

aae_generated_samples_mnist = [
    "/mnt/mfs/mlstorage-experiments/cwx17/72/1c/d434dabfcaec62bb97e5/sample_store.npz", # 1.0 FID: 12.86
    "/mnt/mfs/mlstorage-experiments/cwx17/ef/0c/d4e63c432be94b93a7e5/sample_store.npz", # 0.8 FID: 14.68
    "/mnt/mfs/mlstorage-experiments/cwx17/e3/2c/d4747dc47d248a93a7e5/sample_store.npz", # 0.5 FID: 22.70
    "/mnt/mfs/mlstorage-experiments/cwx17/d3/2c/d4747dc47d249793a7e5/sample_store.npz", # 0.2 FID: 72.00
    "/mnt/mfs/mlstorage-experiments/cwx17/a2/1c/d434dabfcaec5793a7e5/sample_store.npz", # 0.1 FID: 121.24
]

aae_generated_samples_cifar = [
    "/mnt/mfs/mlstorage-experiments/cwx17/b2/1c/d434dabfcaec20a3a7e5/sample_store.npz", # 1.0 FID: 83.03
    "/mnt/mfs/mlstorage-experiments/cwx17/ff/0c/d4e63c432be9e234a7e5/sample_store.npz", # 0.8 FID: 76.13
    "/mnt/mfs/mlstorage-experiments/cwx17/e7/2c/d445f4f80a9f3234a7e5/sample_store.npz", # 0.5 FID: 73.34
    "/mnt/mfs/mlstorage-experiments/cwx17/f7/2c/d445f4f80a9fb774a7e5/sample_store.npz", # 0.2 FID: 114.92
    "/mnt/mfs/mlstorage-experiments/cwx17/f3/2c/d4747dc47d241774a7e5/sample_store.npz", # 0.1 FID: 169.71
]

vae_generated_samples_mnist = [
    "/mnt/mfs/mlstorage-experiments/cwx17/98/2c/d445f4f80a9fcb3da7e5/sample_store.npz", # 1.0 FID: 13.72
    "/mnt/mfs/mlstorage-experiments/cwx17/c4/2c/d4747dc47d24fd3da7e5/sample_store.npz", # 0.8 FID: 24.39
    "/mnt/mfs/mlstorage-experiments/cwx17/60/1c/d4e63c432be9f56da7e5/sample_store.npz", # 0.5 FID: 66.36
    "/mnt/mfs/mlstorage-experiments/cwx17/33/1c/d434dabfcaec966da7e5/sample_store.npz", # 0.2 FID: 129.60
    "/mnt/mfs/mlstorage-experiments/cwx17/d4/2c/d4747dc47d24676da7e5/sample_store.npz", # 0.1 FID: 165.81

]

vae_generated_samples_cifar = [
    "/mnt/mfs/mlstorage-experiments/cwx17/a4/2c/d4747dc47d24321da7e5/sample_store.npz", # 1.0 FID: 133.11
    "/mnt/mfs/mlstorage-experiments/cwx17/b4/2c/d4747dc47d24be2da7e5/sample_store.npz", # 0.8 FID: 135.64
    "/mnt/mfs/mlstorage-experiments/cwx17/40/1c/d4e63c432be9001da7e5/sample_store.npz", # 0.5 FID: 169.75
    "/mnt/mfs/mlstorage-experiments/cwx17/94/2c/d4747dc47d24501da7e5/sample_store.npz", # 0.2 FID: 326.83
    "/mnt/mfs/mlstorage-experiments/cwx17/84/2c/d4747dc47d24dc0da7e5/sample_store.npz", # 0.1 FID: 365.31
]

def draw(list_num):
    idtn=['1.0','0.8','0.5','0.2','0.1']
    if list_num == 0:
        print('++vaepp mnist')
        pairs=[]
        labels=[]
        for idx,path in enumerate(vaepp_generated_samples_mnist):
            res = compute(path,'mnist',savename=f'{idtn[idx]}',savedir=None)
            pairs.append(res)
            labels.append(f'vaepp-mnist-{idtn[idx]}')
            utils.plot(pairs,labels)
    elif list_num == 1:
        print('++vaepp cifar')
        pairs=[]
        labels=[]
        for idx,path in enumerate(vaepp_generated_samples_cifar10):
            res = compute(path,'cifar10',savename=f'{idtn[idx]}',savedir=None)
            pairs.append(res)
            labels.append(f'vaepp-cifar10-{idtn[idx]}')
            utils.plot(pairs,labels)
    elif list_num ==2:
        print('++aae mnist')
        pairs=[]
        labels=[]
        for idx,path in enumerate(aae_generated_samples_mnist):
            res = compute(path,'mnist',savename=f'{idtn[idx]}',savedir=None)
            pairs.append(res)
            labels.append(f'aae-mnist-{idtn[idx]}')
            utils.plot(pairs,labels)
    elif list_num ==3:
        print('++aae cifar')
        pairs=[]
        labels=[]
        for idx,path in enumerate(aae_generated_samples_cifar):
            res = compute(path,'cifar10',savename=f'{idtn[idx]}',savedir=None)
            pairs.append(res)
            labels.append(f'aae-cifar10-{idtn[idx]}')
            utils.plot(pairs,labels)
    elif list_num ==4:
        print('++vae mnist')
        pairs=[]
        labels=[]
        for idx,path in enumerate(vae_generated_samples_mnist):
            res = compute(path,'mnist',savename=f'{idtn[idx]}',savedir=None)
            pairs.append(res)
            labels.append(f'vae-mnist-{idtn[idx]}')
            utils.plot(pairs,labels)
    elif list_num ==5:
        print('++vae mnist')
        pairs=[]
        labels=[]
        for idx,path in enumerate(vae_generated_samples_cifar):
            res = compute(path,'cifar10',savename=f'{idtn[idx]}',savedir=None)
            pairs.append(res)
            labels.append(f'vae-cifar10-{idtn[idx]}')
            utils.plot(pairs,labels)
    else:
        print('list_num should be in [0,5]')


if __name__ == '__main__':
    mnist_test='/mnt/mfs/mlstorage-experiments/cwx17/ae/1c/d4747dc47d24cf35d1e5/sample_store.npz'
    cifar_test='/mnt/mfs/mlstorage-experiments/cwx17/f5/1c/d445f4f80a9f68b140e5/sample_store.npz'
    print('++vaepp')
    idtn=['1.0','0.8','0.5','0.2','0.1']
    for id,path in enumerate(vaepp_generated_samples_mnist):
        compute(path,'mnist',savename=f'{idtn[id]}',savedir='vaepp-mnist')
    for path in vaepp_generated_samples_cifar10:
        compute(path,'cifar10',savename=f'{idtn[id]}',savedir='vaepp-cifar10')
    print('++aae')
    for path in aae_generated_samples_mnist:
        compute(path,'mnist',savename=f'{idtn[id]}',savedir='aae-mnist')
    for path in aae_generated_samples_cifar:
        compute(path,'cifar10',savename=f'{idtn[id]}',savedir='aae-cifar10')
    print('++vae')
    for path in vae_generated_samples_mnist:
        compute(path,'mnist',savename=f'{idtn[id]}',savedir='vae-mnist')
    for path in vae_generated_samples_cifar:
        compute(path,'cifar10',savename=f'{idtn[id]}',savedir='vae-cifar10')
