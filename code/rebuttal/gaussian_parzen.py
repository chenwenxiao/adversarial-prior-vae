import numpy as np
import tfsnippet as spt
from scipy.misc import logsumexp

gen_images_path = "/mnt/mfs/mlstorage-experiments/cwx17/f5/1c/d445f4f80a9f68b140e5/sample_store.npz"
pack = np.load(gen_images_path)
gen_images = pack['mala_img']

(_, real_images), (__, ___) = spt.datasets.load_cifar10(x_shape=(32, 32, 3))

print(gen_images.shape, real_images.shape)

log_prob_list = []
for x in real_images:
    log_prob = 0.5 * np.sum(-0.5 * np.log(np.pi * 2) + (x - gen_images) ** 2, axis=(-1, -2, -3))
    print(log_prob.shape)
    log_prob = logsumexp(log_prob) - np.log(len(log_prob))
    log_prob_list.append(log_prob)

print(np.mean(log_prob_list))

