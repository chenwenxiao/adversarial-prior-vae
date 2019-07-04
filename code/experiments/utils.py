'''
add get_inception_score_tf and get_fid_tf

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
from tensorflow.contrib.gan.python.eval import inception_score, frechet_inception_distance, preprocess_image

from tensorflow.contrib.gan.python.eval import get_graph_def_from_url_tarball

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

INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score.pb'
graph_def = get_graph_def_from_url_tarball(INCEPTION_URL, INCEPTION_FROZEN_GRAPH,
                                           '/home/cwx17/' + os.path.basename(INCEPTION_URL))

# Run images through Inception.
inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, None, None, 3])
activations1 = tf.placeholder(tf.float32, [None, None], name='activations1')
activations2 = tf.placeholder(tf.float32, [None, None], name='activations2')
fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1, activations2)

'''
get_fid_tf and get_inception_score_tf,
images form must be tensor of tf as 
"[batch, height, width]"(for 1 channel) 
or "[batch, height, width, channels]"(for 3 channel)
value of image must be in [0,255]
'''


def get_fid_tf(real_img, sample_img):
    real_img = preprocess_image(real_img)
    sample_img = preprocess_image(sample_img)
    real_single = real_img.shape.ndim == 3
    sample_single = sample_img.shape.ndim == 3
    if real_single and sample_single:
        sample_img = tf.concat([sample_img, sample_img, sample_img], 3)
        real_img = tf.concat([real_img, real_img, real_img], 3)

    FID = frechet_inception_distance(real_img, sample_img)
    return FID


def get_inception_score_tf(sample_img):
    sample_img = preprocess_image(sample_img)
    sample_single = sample_img.shape.ndim == 3
    if sample_single:
        sample_img = tf.concat([sample_img, sample_img, sample_img], 3)

    IS = inception_score(sample_img)
    return IS


get_fid = get_fid_tf
get_inception_score = get_inception_score_tf

if __name__ == '__main__':
    import tfsnippet as spt

    (train_x, train_y), (test_x, test_y) = spt.datasets.load_cifar10(channels_last=False)
    print(get_inception_score_tf(test_x))
    print(get_fid_tf(test_x, test_x))
