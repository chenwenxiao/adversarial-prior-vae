from tensorflow.contrib.gan.python.eval.python.classifier_metrics_impl \
    import inception_score, frechet_inception_distance, preprocess_image, get_graph_def_from_url_tarball
from tensorflow.contrib.gan.python.eval.python.classifier_metrics_impl \
    import classifier_score, frechet_classifier_distance, run_image_classifier
from tensorflow.contrib.gan.python.eval.python.classifier_metrics_impl \
    import INCEPTION_DEFAULT_IMAGE_SIZE, INCEPTION_INPUT, INCEPTION_OUTPUT, _kl_divergence

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
import tfsnippet as spt
import matplotlib.pyplot as plt

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
    print('debug/real_img type')
    print(real_img)
    real_img = tf.convert_to_tensor(real_img)
    real_img = preprocess_image(real_img)
    print(real_img)
    print('debug/sample_img type')
    print(sample_img)
    sample_img = tf.convert_to_tensor(sample_img)
    sample_img = preprocess_image(sample_img)
    print(sample_img)
    real_single = (real_img.shape.ndims == 3)
    sample_single = (sample_img.shape.ndims == 3)
    if real_single and sample_single:
        sample_img = tf.concat([sample_img, sample_img, sample_img], 3)
        real_img = tf.concat([real_img, real_img, real_img], 3)

    ll = real_img.shape[0]
    print('debug/ll')
    print(ll)

    pbs = 4
    while ll % pbs:
        pbs += 1

    FID = frechet_inception_distance(real_img, sample_img, num_batches=ll // pbs)
    return FID


from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def get_inception_score_tf(sample_img):
    print('debug/sample_img type')
    print(sample_img.shape)
    print(sample_img)
    sample_img = tf.convert_to_tensor(sample_img)
    sample_img = preprocess_image(sample_img)
    print(sample_img.shape)
    print(sample_img)

    ll = sample_img.shape[0]
    print('debug/ll')
    print(ll)
    print('debug/sample_img 2')
    print(sample_img.shape)

    pbs = 10
    while True:
        print('debug/pbs', pbs)
        if ll % pbs == 0:
            break
        else:
            pbs -= 1

    print('debug/batch_size')
    print(pbs)

    IS = classifier_score(sample_img, functools.partial(
        run_image_classifier,
        graph_def=graph_def,
        input_tensor=INCEPTION_INPUT,
        output_tensor=INCEPTION_OUTPUT, ), num_batches=ll // 100)

    # generated_images_list = array_ops.split(
    #     sample_img, num_or_size_splits=ll//pbs)
    #
    # # Compute the classifier splits using the memory-efficient `map_fn`.
    # logits = functional_ops.map_fn(
    #     fn=functools.partial(
    #         run_image_classifier,
    #         graph_def=graph_def,
    #         input_tensor=INCEPTION_INPUT,
    #         output_tensor=INCEPTION_OUTPUT,),
    #     elems=array_ops.stack(generated_images_list),
    #     parallel_iterations=1,
    #     back_prop=False,
    #     swap_memory=True,
    #     name='RunClassifier')
    # logits = array_ops.concat(array_ops.unstack(logits), 0)
    # print('logits',logits)
    # logits.shape.assert_has_rank(2)
    #
    # # Use maximum precision for best results.
    # logits_dtype = logits.dtype
    # if logits_dtype != dtypes.float64:
    #     logits = math_ops.to_double(logits)
    #
    # p = nn_ops.softmax(logits)
    # q = math_ops.reduce_mean(p, axis=0)
    # kl = _kl_divergence(p, logits, q)
    # print('pqkl',p,q,kl)
    # kl.shape.assert_has_rank(1)
    #
    # sess = tf.Session()
    # sess.run(tf.Print(kl, [kl]))
    #
    # log_score = math_ops.reduce_mean(kl)
    # print('log_score',log_score)
    # final_score = math_ops.exp(log_score)
    # print('final',final_score)
    # if logits_dtype != dtypes.float64:
    #     final_score = math_ops.cast(final_score, logits_dtype)
    # IS = final_score

    print('debug/end IS')
    IS = IS.shape
    print('debug/end shape')

    return IS


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
        fn=functools.partial(tfgan.eval.run_inception, graph_def=graph_def, output_tensor='pool_3:0'),
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


def get_fid_tsc(images1, images2):
    images1 = images1.transpose(0, 3, 1, 2)
    images2 = images2.transpose(0, 3, 1, 2)
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


def inception_logits(images=inception_images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    logits = functional_ops.map_fn(
        fn=functools.partial(tfgan.eval.run_inception, graph_def=graph_def, output_tensor='logits:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits


logits = inception_logits()


def get_inception_probs(inps):
    n_batches = len(inps) // BATCH_SIZE
    preds = np.zeros([n_batches * BATCH_SIZE, 1000], dtype=np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255. * 2 - 1
        h = logits.eval({inception_images: inp})[:, :1000]
        # print(f'h{i}',h)
        preds[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = h
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds


def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def get_inception_score(images, splits=10):
    images = images.transpose(0, 3, 1, 2)
    assert (type(images) == np.ndarray)
    assert (len(images.shape) == 4)
    assert (images.shape[1] == 3)
    assert (np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'
    print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits))
    start_time = time.time()

    preds = get_inception_probs(images)
    mean, std = preds2score(preds, splits)
    print('Inception Score calculation time: %f s' % (time.time() - start_time))
    return mean, std  # Reference values: 11.34 for 49984 CIFAR-10 training set images, or mean=11.31, std=0.08 if in 10 splits.


from code.experiment.ttur_fid import create_inception_graph, calculate_frechet_distance \
    , calculate_activation_statistics, get_activations

import scipy


def get_fid_ttur(sample_images, real_images):
    # loads all images into memory (this might require a lot of RAM!)
    # print("load images..", end=" " , flush=True)
    # image_list = glob.glob(os.path.join(data_path, '*.jpg'))
    # images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
    # print("%d images found and loaded" % len(images))

    print("create inception graph..", end=" ", flush=True)
    # create_inception_graph(inception_path)  # load the graph into the current TF graph
    tf.import_graph_def(graph_def, name='FID_Inception_Net')
    print("ok")

    print("calculte FID stats..", end=" ", flush=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu_gen, sigma_gen = calculate_activation_statistics(sample_images, sess, batch_size=100)
        mu_real, sigma_real = calculate_activation_statistics(real_images, sess, batch_size=100)
        # np.savez_compressed(output_path, mu=mu, sigma=sigma)
        fid_value = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    print("finished")
    return fid_value


def get_fid_google(sample, real):
    """Returns the FID based on activations.

    Args:
      fake_activations: NumPy array with fake activations.
      real_activations: NumPy array with real activations.
    Returns:
      A float, the Frechet Inception Distance.
    """
    tf.import_graph_def(graph_def, name='FID_Inception_Net')
    with tf.Session() as sess:
        fake_activations = get_activations(sample, sess)
        real_activations = get_activations(real, sess)
        fake_activations = tf.convert_to_tensor(fake_activations)
        real_activations = tf.convert_to_tensor(real_activations)
        fid = tfgan.eval.frechet_classifier_distance_from_activations(
            real_activations=real_activations,
            generated_activations=fake_activations)
        fid = sess.run(fid)
    return fid

def inception_transform(inputs):
    with tf.control_dependencies([
        tf.assert_greater_equal(inputs, 0.0),
        tf.assert_less_equal(inputs, 255.0)]):
        inputs = tf.identity(inputs)
    preprocessed_inputs = tf.map_fn(
        fn=tfgan.eval.preprocess_image,
        elems=inputs,
        back_prop=False)
    return tfgan.eval.run_inception(
        preprocessed_inputs,
        graph_def=graph_def,
        output_tensor=["pool_3:0", "logits:0"])


def inception_transform_np(inputs, batch_size):
    with tf.Session(graph=tf.Graph()) as sess:
        inputs_placeholder = tf.placeholder(
            dtype=tf.float32, shape=[None] + list(inputs[0].shape))
        features_and_logits = inception_transform(inputs_placeholder)
        features = []
        logits = []
        num_batches = int(np.ceil(inputs.shape[0] / batch_size))
        for i in range(num_batches):
            input_batch = inputs[i * batch_size:(i + 1) * batch_size]
            x = sess.run(
                features_and_logits, feed_dict={inputs_placeholder: input_batch})
            features.append(x[0])
            logits.append(x[1])
        features = np.vstack(features)
        logits = np.vstack(logits)
        return features, logits

def get_fid_compare_gan(sample, real):
    """Returns the FID
    
    basically this function has same 
    logical reesult with get_fid_google
    this function is same with compare_gan
    at https://github.com/google/compare_gan/blob/master/compare_gan/eval_utils.py

    Returns:
      A float, the Frechet Inception Distance.
    """
    tf.import_graph_def(graph_def, name='FID_Inception_Net')
    with tf.Session() as sess:
        (fake_activations,_) = inception_transform_np(sample, 30)
        (real_activations,_) = inception_transform_np(real, 30)
        fake_activations = tf.convert_to_tensor(fake_activations)
        real_activations = tf.convert_to_tensor(real_activations)
        fid = tfgan.eval.frechet_classifier_distance_from_activations(
            real_activations=real_activations,
            generated_activations=fake_activations)
        fid = sess.run(fid)
    return fid


def get_mean_cov(ims, sess, batch_size=100):
    n, c, w, h = ims.shape
    print('Batch size:', batch_size)
    print('Total number of images:', n)
    pred = get_activations(ims, sess)
    mean = np.mean(pred, axis=0)
    cov = np.cov(pred)
    return mean, cov


def FID(m0, c0, m1, c1):
    ret = 0
    ret += np.sum((m0 - m1) ** 2)
    ret += np.trace(c0 + c1 - 2.0 * scipy.linalg.sqrtm(np.dot(c0, c1)))
    return np.real(ret)


def get_fid_sngan(sample, real, batchsize=100):
    """Frechet Inception Distance proposed by https://arxiv.org/abs/1706.08500"""

    tf.import_graph_def(graph_def, name='FID_Inception_Net')

    with tf.Session() as sess:
        m1, c1 = get_mean_cov(sample, sess);
        m2, c2 = get_mean_cov(real, sess);
    fid = FID(m1, c1, m2, c2)
    return fid


get_fid = get_fid_compare_gan


#adapted from https://github.com/kynkaat/improved-precision-and-recall-metric

import os
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import uuid
import code.experiment.dnnlib as dnnlib
import code.experiment.dnnlib.tflib.tfutil as tfutil

from skimage import transform
from typing import Any, List, Tuple, Union
sys.path.append('experiment')
from code.experiment.pnr import knn_precision_recall_features

def init_tf(random_seed=1234):
    """Initialize TF."""
    print('Initializing TensorFlow...\n')
    np.random.seed(random_seed)
    tfutil.init_tf({'graph_options.place_pruned_graph': True,
                    'gpu_options.allow_growth': True})

def V_pre(images,height=224,width=224,scope=None):
    '''
    '''
    if len(images.shape)==3:
        images=np.array((images,images,images)).transpose(1,0,2,3)
    else:
        images=images.transpose(0,3,1,2)
    return images

def initialize_feature_extractor():
    """Load VGG-16 network pickle (returns features from FC layer with shape=(n, 4096))."""
    print('Initializing VGG-16 model...')
    url = 'https://drive.google.com/uc?id=1fk6r8vetqpRShtEODXm9maDytbMkHLfa' # vgg16.pkl
    with dnnlib.util.open_url(url, cache_dir=os.path.join(os.path.dirname(__file__), '_cache')) as f:
        _, _, net = pickle.load(f)
    # local='code/experiment/_cache/q.pkl'
    # with open(local, "rb") as f:
    #     _, _, net = pickle.load(f)
    return net

def precision_recall(real_images,gen_images,num_images,batch_size=20,num_gpus=1):
    '''accept two numpy arrat of images'''
    # reshape
    real_images = V_pre(real_images)
    gen_images = V_pre(gen_images)

    print(real_images.shape)
    init_tf()
    feature_net=initialize_feature_extractor()

    print('compute for real images...')
    real_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)
    for begin in range(0, num_images, batch_size):
        end = min(begin + batch_size, num_images)
        real_batch = real_images[begin:end]
        real_features[begin:end] = feature_net.run(real_batch, num_gpus=num_gpus, assume_frozen=True)

    # Calculate VGG-16 features for generated images.
    print('compute for generating images')
    gen_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)
    for begin in range(0, num_images, batch_size):
        end = min(begin + batch_size, num_images)
        gen_batch = gen_images[begin:end]
        gen_features[begin:end] = feature_net.run(gen_batch, num_gpus=num_gpus, assume_frozen=True)

    print(gen_features.shape,real_features.shape)
    newway = compute_prd_from_embedding(gen_features,real_features)
    # state = knn_precision_recall_features(real_features, gen_features, num_gpus=num_gpus)
    state={'precision':'none','recall':'none'}
    return (state['precision'], state['recall'], newway[0],newway[1])

import sklearn.cluster


def compute_prd(eval_dist, ref_dist, num_angles=1001, epsilon=1e-10):
    """Computes the PRD curve for discrete distributions.

    This function computes the PRD curve for the discrete distribution eval_dist
    with respect to the reference distribution ref_dist. This implements the
    algorithm in [arxiv.org/abs/1806.2281349]. The PRD will be computed for an
    equiangular grid of num_angles values between [0, pi/2].

    Args:
      eval_dist: 1D NumPy array or list of floats with the probabilities of the
                 different states under the distribution to be evaluated.
      ref_dist: 1D NumPy array or list of floats with the probabilities of the
                different states under the reference distribution.
      num_angles: Number of angles for which to compute PRD. Must be in [3, 1e6].
                  The default value is 1001.
      epsilon: Angle for PRD computation in the edge cases 0 and pi/2. The PRD
               will be computes for epsilon and pi/2-epsilon, respectively.
               The default value is 1e-10.

    Returns:
      precision: NumPy array of shape [num_angles] with the precision for the
                 different ratios.
      recall: NumPy array of shape [num_angles] with the recall for the different
              ratios.

    Raises:
      ValueError: If not 0 < epsilon <= 0.1.
      ValueError: If num_angles < 3.
    """

    if not (epsilon > 0 and epsilon < 0.1):
        raise ValueError('epsilon must be in (0, 0.1] but is %s.' % str(epsilon))
    if not (num_angles >= 3 and num_angles <= 1e6):
        raise ValueError('num_angles must be in [3, 1e6] but is %d.' % num_angles)

    # Compute slopes for linearly spaced angles between [0, pi/2]
    angles = np.linspace(epsilon, np.pi/2 - epsilon, num=num_angles)
    slopes = np.tan(angles)

    # Broadcast slopes so that second dimension will be states of the distribution
    slopes_2d = np.expand_dims(slopes, 1)

    # Broadcast distributions so that first dimension represents the angles
    ref_dist_2d = np.expand_dims(ref_dist, 0)
    eval_dist_2d = np.expand_dims(eval_dist, 0)

    # Compute precision and recall for all angles in one step via broadcasting
    precision = np.minimum(ref_dist_2d*slopes_2d, eval_dist_2d).sum(axis=1)
    recall = precision / slopes

    # handle numerical instabilities leaing to precision/recall just above 1
    max_val = max(np.max(precision), np.max(recall))
    if max_val > 1.001:
        raise ValueError('Detected value > 1.001, this should not happen.')
    precision = np.clip(precision, 0, 1)
    recall = np.clip(recall, 0, 1)

    return precision, recall


def _cluster_into_bins(eval_data, ref_data, num_clusters):
    """Clusters the union of the data points and returns the cluster distribution.

    Clusters the union of eval_data and ref_data into num_clusters using minibatch
    k-means. Then, for each cluster, it computes the number of points from
    eval_data and ref_data.

    Args:
      eval_data: NumPy array of data points from the distribution to be evaluated.
      ref_data: NumPy array of data points from the reference distribution.
      num_clusters: Number of cluster centers to fit.

    Returns:
      Two NumPy arrays, each of size num_clusters, where i-th entry represents the
      number of points assigned to the i-th cluster.
    """

    cluster_data = np.vstack([eval_data, ref_data])
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters, n_init=10)
    labels = kmeans.fit(cluster_data).labels_

    eval_labels = labels[:len(eval_data)]
    ref_labels = labels[len(eval_data):]

    eval_bins = np.histogram(eval_labels, bins=num_clusters,
                             range=[0, num_clusters], density=True)[0]
    ref_bins = np.histogram(ref_labels, bins=num_clusters,
                            range=[0, num_clusters], density=True)[0]
    return eval_bins, ref_bins


def compute_prd_from_embedding(eval_data, ref_data, num_clusters=20,
                               num_angles=1001, num_runs=10,
                               enforce_balance=True):
    """Computes PRD data from sample embeddings.

    The points from both distributions are mixed and then clustered. This leads
    to a pair of histograms of discrete distributions over the cluster centers
    on which the PRD algorithm is executed.

    The number of points in eval_data and ref_data must be equal since
    unbalanced distributions bias the clustering towards the larger dataset. The
    check can be disabled by setting the enforce_balance flag to False (not
    recommended).

    Args:
      eval_data: NumPy array of data points from the distribution to be evaluated.
      ref_data: NumPy array of data points from the reference distribution.
      num_clusters: Number of cluster centers to fit. The default value is 20.
      num_angles: Number of angles for which to compute PRD. Must be in [3, 1e6].
                  The default value is 1001.
      num_runs: Number of independent runs over which to average the PRD data.
      enforce_balance: If enabled, throws exception if eval_data and ref_data do
                       not have the same length. The default value is True.

    Returns:
      precision: NumPy array of shape [num_angles] with the precision for the
                 different ratios.
      recall: NumPy array of shape [num_angles] with the recall for the different
              ratios.

    Raises:
      ValueError: If len(eval_data) != len(ref_data) and enforce_balance is set to
                  True.
    """

    if enforce_balance and len(eval_data) != len(ref_data):
        raise ValueError(
            'The number of points in eval_data %d is not equal to the number of '
            'points in ref_data %d. To disable this exception, set enforce_balance '
            'to False (not recommended).' % (len(eval_data), len(ref_data)))

    eval_data = np.array(eval_data, dtype=np.float64)
    ref_data = np.array(ref_data, dtype=np.float64)
    precisions = []
    recalls = []
    for _ in range(num_runs):
        eval_dist, ref_dist = _cluster_into_bins(eval_data, ref_data, num_clusters)
        precision, recall = compute_prd(eval_dist, ref_dist, num_angles)
        precisions.append(precision)
        recalls.append(recall)
    precision = np.mean(precisions, axis=0)
    recall = np.mean(recalls, axis=0)
    return precision, recall

def plot(precision_recall_pairs, labels=None, out_path=None,
         legend_loc='lower left', dpi=300,name=''):
    """Plots precision recall curves for distributions.

    Creates the PRD plot for the given data and stores the plot in a given path.

    Args:
      precision_recall_pairs: List of prd_data to plot. Each item in this list is
                              a 2D array of precision and recall values for the
                              same number of ratios.
      labels: Optional list of labels of same length as list_of_prd_data. The
              default value is None.
      out_path: Output path for the resulting plot. If None, the plot will be
                opened via plt.show(). The default value is None.
      legend_loc: Location of the legend. The default value is 'lower left'.
      dpi: Dots per inch (DPI) for the figure. The default value is 150.

    Raises:
      ValueError: If labels is a list of different length than list_of_prd_data.
    """

    if labels is not None and len(labels) != len(precision_recall_pairs):
        raise ValueError(
            'Length of labels %d must be identical to length of '
          'precision_recall_pairs %d.'
          % (len(labels), len(precision_recall_pairs)))

    fig = plt.figure(figsize=(3.5, 3.5), dpi=dpi)
    plot_handle = fig.add_subplot(111)
    plot_handle.tick_params(axis='both', which='major', labelsize=12)

    for i in range(len(precision_recall_pairs)):
        precision, recall = precision_recall_pairs[i]
        label = labels[i] if labels is not None else None
        plt.plot(recall, precision, label=label, alpha=0.5, linewidth=1)

    if labels is not None:
        plt.legend(loc=legend_loc,fontsize=5)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall', fontsize=10)
    plt.ylabel('Precision', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'./prd-{name}', bbox_inches='tight', dpi=dpi)
    plt.close()


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = spt.datasets.load_cifar10(channels_last=True)
    r=train_x[:100]
    g=test_x[:100]
    r=np.array(r)
    g=np.array(g)
    print('r',type(r),r.shape)
    (precision,recall,newp,newr)= precision_recall(r,g,100,20,1)
    print('1 precision',precision,'recall',recall)
    print('2 precision',newp,'\nrecall',newr)
    for i in newp:
        print(i,end='')
    for i in newr:
        print(i,end='')
    '''
    (train_x, train_y), (test_x, test_y) = spt.datasets.load_cifar10(channels_last=True)
    from code.experiment.datasets import celeba
    train_x, validate_x, test_x = celeba.load_celeba()

    x1 = train_x
    x2 = train_x

    if len(x1) > len(x2):
        x1 = x1[:len(x2)]
    else:
        x2 = x2[:len(x1)]

    print("ttur fid's calculating")
    ttur = get_fid_ttur(x1, x2)
    print(f"ttur fid's {ttur}")

    print("sngan fid's calculating")
    sngan = get_fid_sngan(x1, x2)
    print(f"sngan fid's {sngan}")

    print("google fid's calculating")
    google = get_fid_google(x1, x2)
    print(f"google fid's {google}")

    print("compare gan fid's calculating")
    compare_gan = get_fid_compare_gan(x1, x2)
    print(f"compare gan fid's {compare_gan}")

    print(f"compare gan: {compare_gan}\n google: {google}\n sngan: {sngan}\n ttur: {ttur}\n")


    # print(get_inception_score_tf(test_x))
    # print(get_fid_tf(test_x, test_x))
    '''