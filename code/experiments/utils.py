'''
computation of IS and FID is from
https://github.com/tsc2017/inception-score
and
https://github.com/tsc2017/Frechet-Inception-Distance


Interface :
    get_inception_score(images, splits=10) :
        spliting total images into 'splits' group, and compute IS separtely.
        reture two np array, mean and std, with length of 'splits'
    get_fid(image_1,image_2) :
        return one scalar of FID

altering dmm.py to use this
0)  import utils.py
    At dmm_modified.py line 21,22

1)  add a scope of 'scoring'
    In this scope, 'sample_img' and 'reconstruct_img' are defined to acquire image
    At dmm_modified.py line 576:590+1

2)  When epoch arrives at 'max_epoch'
    get image of sample and reconstruct, then compute IS and FID using interface mentioned above
    At dmm_modified.py line 733:789+1


'''

'''
From https://github.com/tsc2017/Frechet-Inception-Distance
Code derived from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
Usage:
    Call get_fid(images1, images2)
Args:
    images1, images2: Numpy arrays with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary. 
    dtype of the images is recommended to be np.uint8 to save CPU memory.
Returns:
    Frechet Inception Distance between the two image distributions.
'''

import tensorflow as tf
import os, sys
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops

tfgan = tf.contrib.gan

session = tf.InteractiveSession()

# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 64

# Run images through Inception.
inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
activations1 = tf.placeholder(tf.float32, [None, None], name='activations1')
activations2 = tf.placeholder(tf.float32, [None, None], name='activations2')
fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1, activations2)


def inception_activations(images=inception_images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    activations = functional_ops.map_fn(
        fn=functools.partial(tfgan.eval.run_inception, output_tensor='pool_3:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations


activations = inception_activations()


def get_inception_activations(inps):
    n_batches = inps.shape[0] // BATCH_SIZE
    act = np.zeros([n_batches * BATCH_SIZE, 2048], dtype=np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255. * 2 - 1
        act[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = activations.eval(feed_dict={inception_images: inp})
    return act


def activations2distance(act1, act2):
    return fcd.eval(feed_dict={activations1: act1, activations2: act2})


def get_fid(images1, images2):
    assert (type(images1) == np.ndarray)
    assert (len(images1.shape) == 4)
    assert (images1.shape[1] == 3)
    assert (np.min(images1[0]) >= 0 and np.max(images1[0]) > 10), 'Image values should be in the range [0, 255]'
    assert (type(images2) == np.ndarray)
    assert (len(images2.shape) == 4)
    assert (images2.shape[1] == 3)
    assert (np.min(images2[0]) >= 0 and np.max(images2[0]) > 10), 'Image values should be in the range [0, 255]'
    assert (images1.shape == images2.shape), 'The two numpy arrays must have the same shape'
    print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(images1)
    act2 = get_inception_activations(images2)
    fid = activations2distance(act1, act2)
    print('FID calculation time: %f s' % (time.time() - start_time))
    return fid


'''
From https://github.com/tsc2017/inception-score
Code derived from https://github.com/openai/improved-gan/blob/master/inception_score/model.py
Args:
    images: A numpy array with values ranging from -1 to 1 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary.
    splits: The number of splits of the images, default is 10.
Returns:
    mean and standard deviation of the inception across the splits.
'''

# Run images through Inception.
inception_images = tf.placeholder(
    tf.float32,
    [BATCH_SIZE, 3, None, None]
)


def inception_logits(images=inception_images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(
        images, num_or_size_splits=num_splits
    )
    logits = functional_ops.map_fn(
        fn=functools.partial(tfgan.eval.run_inception, output_tensor='logits:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier'
    )
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits


logits = inception_logits()


def get_inception_probs(inps):
    preds = []
    n_batches = len(inps) // BATCH_SIZE
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        pred = logits.eval({inception_images: inp})[:, :1000]
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds


def preds2score(preds, splits):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def get_inception_score(images, splits=10):
    assert (type(images) == np.ndarray)
    assert (len(images.shape) == 4)
    assert (images.shape[1] == 3)
    assert (np.max(images[0]) <= 1)
    assert (np.min(images[0]) >= -1)

    start_time = time.time()
    preds = get_inception_probs(images)
    print('Inception Score for %i samples in %i splits' % (preds.shape[0], splits))
    mean, std = preds2score(preds, splits)
    print('Inception Score calculation time: %f s' % (time.time() - start_time))
    return mean, std  # Reference values: 11.34 for 49984 CIFAR-10 training set images, or mean=11.31, std=0.08 if in 10 splits (default).


if __name__ == '__main__':
    import tfsnippet as spt
    (train_x, train_y), (test_x, test_y) = spt.datasets.load_cifar10(channels_last=False, normalize_x=True)
    print(get_inception_score(train_x))