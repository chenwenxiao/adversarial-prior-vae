import numpy as np
import tfsnippet as spt
from scipy.misc import logsumexp
import time

from code.experiment.datasets.omniglot import load_omniglot

gen_images_path = "/mnt/mfs/mlstorage-experiments/cwx17/13/1c/d434dabfcaec409ca7e5/sample_store.npz"
pack = np.load(gen_images_path)
gen_images = pack['mala_img']

(_, __), (real_images, ___) = load_omniglot(x_shape=(28, 28, 1))

gen_images = gen_images[..., 0]
gen_images = np.reshape(gen_images, (-1, 28, 28, 1))
print(gen_images.shape, real_images.shape)
real_images = real_images[:1000]
gen_images = gen_images / 255.0
real_images = real_images / 255.0
sigma = 0.2
log_prob_list = []
for i in range(len(real_images)):
    x = real_images[i]
    print(x.shape)
    log_prob = np.sum(-0.5 * np.log(np.pi * 2) - 0.5 * (x - gen_images) ** 2 / (2 * sigma * sigma) - np.log(sigma),
                      axis=(-1, -2, -3))
    print(log_prob)
    log_prob = logsumexp(log_prob) - np.log(len(log_prob))
    log_prob_list.append(log_prob)
    print(len(log_prob_list), log_prob)

print(np.mean(log_prob_list))
