# -*- coding: utf-8 -*-
import functools
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
import tensorflow as tf
from pprint import pformat

from matplotlib import pyplot
from tensorflow.contrib.framework import arg_scope, add_arg_scope

import tfsnippet as spt
from code.experiments.datasets.omniglot import load_omniglot
from code.experiments.truncated_normal import TruncatedNormal
from tfsnippet import DiscretizedLogistic
from tfsnippet.examples.utils import (MLResults,
                                      save_images_collection,
                                      bernoulli_as_pixel,
                                      bernoulli_flow,
                                      bernoulli_flow,
                                      print_with_title)
from code.experiments.utils import get_inception_score, get_fid
import numpy as np
from scipy.misc import logsumexp

from tfsnippet.preprocessing import UniformNoiseSampler, BernoulliSampler

spt.settings.check_numerics = spt.settings.enable_assertions = False


def _bernoulli_mean(self):
    if not hasattr(self, '_mean'):
        self._mean = tf.sigmoid(self.logits)
    return self._mean


spt.Bernoulli.mean = property(_bernoulli_mean)


class ExpConfig(spt.Config):
    # model parameters
    z_dim = 36
    act_norm = False
    weight_norm = False
    l2_reg = 0.0002
    kernel_size = 3
    shortcut_kernel_size = 1
    batch_norm = True
    nf_layers = 5

    # training parameters
    result_dir = None
    write_summary = True
    max_epoch = 2000
    warm_up_start = 1000
    warm_up_epoch = 1000
    beta = 1e-8
    initial_xi = 0.0
    pull_back_energy_weight = 10.0

    max_step = None
    batch_size = 128
    noise_len = 8
    smallest_step = 5e-5
    initial_lr = 0.0001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    lr_anneal_step_freq = None

    use_flow = False
    use_dg = False
    gradient_penalty_algorithm = 'interpolate'  # both or interpolate
    gradient_penalty_weight = 2
    gradient_penalty_index = 6
    kl_balance_weight = 1.0

    n_critical = 5
    # evaluation parameters
    train_n_pz = 128
    train_n_qz = 1
    test_n_pz = 1000
    test_n_qz = 100
    test_batch_size = 8
    test_epoch_freq = 1000
    plot_epoch_freq = 20
    grad_epoch_freq = 10

    test_fid_n_pz = 5000
    test_x_samples = 1
    log_Z_times = 100000
    log_Z_x_samples = 64

    len_train = 50000
    sample_n_z = 100
    fid_samples = 5000

    use_truncated = True

    epsilon = -20.0
    min_logstd_of_q = -5.0

    @property
    def x_shape(self):
        return (28, 28, 1)

    x_shape_multiple = 28 * 28 * 1


config = ExpConfig()


class EnergyDistribution(spt.Distribution):
    """
    A distribution derived from an energy function `D(x)` and a generator
    function `x = G(z)`, where `p(z) = exp(-xi * D(G(z)) - 0.5 * z^2) / Z`.
    """

    def __init__(self, pz, G, D, log_Z=0., xi=1.0, mcmc_iterator=0, mcmc_alpha=config.smallest_step,
                 mcmc_algorithm='mala',
                 mcmc_space='z', initial_z=None):
        """
        Construct a new :class:`EnergyDistribution`.

        Args:
            pz (spt.Distribution): The base distribution `p(z)`.
            G: The function `x = G(z)`.
            D: The function `D(x)`.
            Z: The partition factor `Z`.
            xi: The weight of energy.
        """
        if not pz.is_continuous:
            raise TypeError('`base_distribution` must be a continuous '
                            'distribution.')

        super(EnergyDistribution, self).__init__(
            dtype=pz.dtype,
            is_continuous=True,
            is_reparameterized=pz.is_reparameterized,
            batch_shape=pz.batch_shape,
            batch_static_shape=pz.get_batch_shape(),
            value_ndims=pz.value_ndims
        )
        log_Z = spt.ops.convert_to_tensor_and_cast(log_Z, dtype=pz.dtype)

        self._pz = pz
        self._G = G
        self._D = D
        self._xi = xi
        with tf.name_scope('log_Z', values=[log_Z]):
            self._log_Z = tf.maximum(log_Z, -20)
        self._mcmc_iterator = mcmc_iterator
        self._mcmc_alpha = mcmc_alpha
        self._initial_z = initial_z

    @property
    def pz(self):
        return self._pz

    @property
    def G(self):
        return self._G

    @property
    def D(self):
        return self._D

    @property
    def xi(self):
        return self._xi

    @property
    def log_Z(self):
        return self._log_Z

    def set_log_Z(self, new_log_Z):
        self._log_Z = new_log_Z
        return new_log_Z

    def log_prob(self, given, group_ndims=0, name=None, y=None):
        given = tf.convert_to_tensor(given)
        with tf.name_scope(name,
                           default_name=spt.utils.get_default_scope_name(
                               'log_prob', self),
                           values=[given]):
            energy = config.pull_back_energy_weight * self.D(self.G(given)) * self.xi + 0.5 * tf.reduce_sum(
                tf.square(given), axis=-1)
            log_px = self.pz.log_prob(given=given, group_ndims=group_ndims)
            log_px.log_energy_prob = -energy - self.log_Z
            log_px.energy = energy
            log_px.pure_energy = self.D(self.G(given))

        return log_px

    def sample(self, n_samples=None, group_ndims=0, is_reparameterized=None,
               compute_density=None, name=None):
        self._validate_sample_is_reparameterized_arg(is_reparameterized)
        if is_reparameterized is None:
            is_reparameterized = self.is_reparameterized

        with tf.name_scope(name, default_name=spt.utils.get_default_scope_name('sample', self)):
            origin_z = self.pz.sample(
                n_samples=n_samples, is_reparameterized=is_reparameterized,
                compute_density=False
            )
            z = origin_z
            if self._initial_z is not None:
                z = self._initial_z
            history_z = []
            history_e_z = []
            history_pure_e_z = []
            history_ratio = []
            for i in range(self._mcmc_iterator):
                e_z, grad_e_z, z_prime, pure_energy_z = self.get_sgld_proposal(z)
                e_z_prime, grad_e_z_prime, _, __ = self.get_sgld_proposal(z_prime)
                history_e_z.append(e_z)
                history_z.append(z)
                history_pure_e_z.append(pure_energy_z)
                log_q_zprime_z = tf.reduce_sum(
                    tf.square(z_prime - z + self._mcmc_alpha * grad_e_z), axis=-1
                )
                log_q_zprime_z *= -1. / (4 * self._mcmc_alpha)

                log_q_z_zprime = tf.reduce_sum(
                    tf.square(z - z_prime + self._mcmc_alpha * grad_e_z_prime), axis=-1
                )
                log_q_z_zprime *= -1. / (4 * self._mcmc_alpha)

                log_ratio_1 = -e_z_prime + e_z  # log [p(z_prime) / p(z)]
                log_ratio_2 = log_q_z_zprime - log_q_zprime_z  # log [q(z | z_prime) / q(z_prime | z)]
                # print(log_ratio_1.mean().item(), log_ratio_2.mean().item())
                ratio = tf.clip_by_value(
                    tf.exp(log_ratio_1 + log_ratio_2), 0.0, 1.0
                )
                history_ratio.append(ratio)
                # print(ratio.mean().item())
                rnd_u = tf.random.uniform(
                    shape=ratio.shape
                )
                mask = tf.cast(tf.less(rnd_u, ratio), tf.float32)
                mask = tf.expand_dims(mask, axis=-1)
                z = (z_prime * mask + z * (1 - mask))

            t = spt.StochasticTensor(
                distribution=self,
                tensor=z,
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized
            )
            t.history_z = tf.stack(history_z)
            t.history_e_z = tf.stack(history_e_z)
            t.history_pure_e_z = tf.stack(history_pure_e_z)
            t.history_ratio = tf.stack(history_ratio)
        return t

    def get_sgld_proposal(self, z):
        energy_z = config.pull_back_energy_weight * self.D(self.G(z)) * self.xi + 0.5 * tf.reduce_sum(tf.square(z),
                                                                                                      axis=-1)
        pure_energy_z = self.D(self.G(z))
        # energy_z = pure_energy_z  # TODO
        grad_energy_z = tf.gradients(energy_z, [z.tensor if hasattr(z, 'tensor') else z])[0]
        grad_energy_z = tf.reshape(grad_energy_z, shape=z.shape)
        eps = tf.random.normal(
            shape=z.shape
        ) * tf.sqrt(self._mcmc_alpha * 2)
        z_prime = z - self._mcmc_alpha * grad_energy_z + eps
        return energy_z, grad_energy_z, z_prime, pure_energy_z


class ExponentialDistribution(spt.Distribution):
    """
    A distribution in R^N, derived from an a mean `\mu`,
    where `p(x) = exp(-\|x - \mu\| / \beta) / Z(\beta)`,
    Z(\beta) = 2 * \pi^(N/2) * \Gamma(N) / \Gamma(N/2) * \beta^N.
    """

    def __init__(self, mean, beta, D):
        """
        Construct a new :class:`EnergyDistribution`.

        Args:
            mean: The mean of ExponentialDistribution.
            beta: The beta of ExponentialDistribution.
        """
        beta = spt.ops.convert_to_tensor_and_cast(beta, dtype=tf.float32)
        _ = spt.Normal(mean, logstd=beta)
        super(ExponentialDistribution, self).__init__(
            dtype=tf.float32,
            is_continuous=True,
            is_reparameterized=True,
            batch_shape=_.batch_shape,
            batch_static_shape=_.get_batch_shape(),
            value_ndims=_.value_ndims
        )
        self._beta = beta
        self._mean = mean
        self._D = D
        self._N = 1
        for i in config.x_shape:
            self._N *= i
        self._log_Z = np.log(2) + self.N / 2.0 * np.log(np.pi) + self.N * tf.log(self.beta)
        bias = 0.0
        for i in range(1, self.N):
            bias += np.log(i)
        for i in range(self.N - 2, 0, -2):
            bias -= np.log(i / 2.0)
        if self.N % 2 == 1:
            bias -= 0.5 * np.log(np.pi)
        self._log_Z += bias

    @property
    def beta(self):
        return self._beta

    @property
    def mean(self):
        return self._mean

    @property
    def D(self):
        return self._D

    @property
    def N(self):
        return self._N

    @property
    def log_Z(self):
        return self._log_Z

    def log_prob(self, given, group_ndims=0, name=None):
        given = tf.convert_to_tensor(given)
        with tf.name_scope(name,
                           default_name=spt.utils.get_default_scope_name('log_prob', self),
                           values=[given]):
            log_px = -tf.sqrt(
                tf.reduce_sum(tf.square(given - self.mean), axis=tf.range(-group_ndims, 0))) / self.beta - self.log_Z
            log_px.energy = self.D(given)
            log_px.mean_energy = self.D(self.mean)

        return log_px

    def sample(self, n_samples=None, group_ndims=0, is_reparameterized=None,
               compute_density=None, name=None):
        self._validate_sample_is_reparameterized_arg(is_reparameterized)
        if is_reparameterized is None:
            is_reparameterized = self.is_reparameterized

        with tf.name_scope(name, default_name=spt.utils.get_default_scope_name('sample', self)):
            t = spt.StochasticTensor(
                distribution=self,
                tensor=self.mean,
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized
            )
        return t


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


@add_arg_scope
def batch_norm(inputs, training=False, scope=None):
    return tf.layers.batch_normalization(inputs, training=training, name=scope)


@add_arg_scope
def dropout(inputs, training=False, scope=None):
    print(inputs, training)
    return spt.layers.dropout(inputs, rate=0.2, training=training, name=scope)


def get_z_moments(z, value_ndims, name=None):
    """
    Get the per-dimensional mean and variance of `z`.

    Args:
        z (Tensor): The `z` tensor.
        value_ndims (int): Number of value dimensions, must be `1` or `3`.
            `1` indicates `z` is a dense latent variable, otherwise `3`
            indicates `z` is a convolutional latent variable.

    Returns:
        (tf.Tensor, tf.Tensor): The `(mean, variance)` of `z`.
    """
    value_ndims = spt.utils.validate_enum_arg(
        'value_ndims', value_ndims, [1, 3])

    if value_ndims == 1:
        z = spt.utils.InputSpec(shape=['...', '*']).validate('z', z)
    else:
        z = spt.utils.InputSpec(shape=['...', '?', '?', '*']).validate('z', z)

    with tf.name_scope(name, default_name='get_z_moments', values=[z]):
        rank = len(spt.utils.get_static_shape(z))
        if value_ndims == 1:
            axes = list(range(0, rank - 1))
        else:
            axes = list(range(0, rank - 3))
        mean, variance = tf.nn.moments(z, axes=axes)
        return mean, variance


@add_arg_scope
@spt.global_reuse
def q_net(x, posterior_flow, observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg), ):
        h_x = tf.to_float(x)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (28, 28, 16)
        # h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_2')  # output: (14, 14, 32)
        # h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_3')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, strides=2, scope='level_4')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, strides=2, scope='level_6')  # output: (7, 7, 64)
        # h_x = spt.layers.resnet_conv2d_block(h_x, 512, strides=2, scope='level_8')  # output: (7, 7, 64)

    # sample z ~ q(z|x)
    h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
    # x = spt.ops.reshape_tail(x, ndims=3, shape=[-1])
    # h_x = tf.concat([h_x, x], axis=-1)
    z_mean = spt.layers.dense(h_x, config.z_dim, scope='z_mean', kernel_initializer=tf.zeros_initializer())
    z_logstd = spt.layers.dense(h_x, config.z_dim, scope='z_logstd', kernel_initializer=tf.zeros_initializer())

    # sample z ~ q(z|x)
    if config.use_truncated:
        z_distribution = TruncatedNormal(mean=z_mean,
                                         logstd=spt.ops.maybe_clip_value(z_logstd, min_val=config.min_logstd_of_q))
        print(z_distribution.batch_shape)
    else:
        z_distribution = spt.Normal(mean=z_mean,
                                    logstd=spt.ops.maybe_clip_value(z_logstd, min_val=config.min_logstd_of_q))
    if config.use_flow:

        z_distribution = spt.FlowDistribution(
            z_distribution,
            posterior_flow
        )
        z = net.add('z', z_distribution, n_samples=n_z)
    else:
        z = net.add('z', z_distribution, n_samples=n_z, group_ndims=1)

    return net


__log_Z = None


@spt.global_reuse
def get_log_Z():
    global __log_Z
    if __log_Z is None:
        __log_Z = spt.ScheduledVariable('log_Z', dtype=tf.float32, initial_value=1., model_var=True)
    return __log_Z


@add_arg_scope
@spt.global_reuse
def G_theta(z):
    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_deconv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = spt.layers.dense(z, 64 * config.x_shape[0] // 4 * config.x_shape[1] // 4, scope='level_0',
                               normalizer_fn=None)
        h_z = spt.ops.reshape_tail(
            h_z,
            ndims=1,
            shape=(config.x_shape[0] // 4, config.x_shape[1] // 4, 64)
        )
        # h_z = spt.layers.resnet_deconv2d_block(h_z, 512, strides=2, scope='level_2')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 64, strides=2, scope='level_3')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32, strides=2, scope='level_5')  # output: (14, 14, 32)
        # h_z = spt.layers.resnet_deconv2d_block(h_z, 32, scope='level_6')  # output:
        # h_z = spt.layers.resnet_deconv2d_block(h_z, 32, scope='level_7')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16, scope='level_8')  # output: (28, 28, 16)
    x_mean = spt.layers.conv2d(
        h_z, config.x_shape[-1], (1, 1), padding='same', scope='feature_map_mean_to_pixel',
        kernel_initializer=tf.zeros_initializer(), activation_fn=tf.nn.tanh
    )
    return x_mean


@add_arg_scope
@spt.global_reuse
def G_omega(z):
    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=True,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = z
        h_z = spt.layers.dense(h_z, 40)
        h_z = spt.layers.dense(h_z, 40)
        h_z = spt.layers.dense(h_z, 40)
        h_z = spt.layers.dense(h_z, 40)
        h_z = spt.layers.dense(h_z, 40)
        h_z = spt.layers.dense(h_z, config.z_dim)
    x_mean = h_z
    return x_mean


@add_arg_scope
@spt.global_reuse
def D_psi(x, y=None):
    # if y is not None:
    #     return D_psi(y) + 0.1 * tf.sqrt(tf.reduce_sum((x - y) ** 2, axis=tf.range(-len(config.x_shape), 0)))
    normalizer_fn = None
    # x = tf.round(256.0 * x / 2 + 127.5)
    # x = (x - 127.5) / 256.0 * 2
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=spectral_norm if config.gradient_penalty_algorithm == 'spectral' else None,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = tf.to_float(x)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (28, 28, 16)
        # h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_2')  # output: (14, 14, 32)
        # h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_3')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, strides=2, scope='level_4')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, strides=2, scope='level_6')  # output: (7, 7, 64)
        # h_x = spt.layers.resnet_conv2d_block(h_x, 512, strides=2, scope='level_8')  # output: (7, 7, 64)

        h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
        h_x = spt.layers.dense(h_x, 64, scope='level_-2')
    # sample z ~ q(z|x)
    h_x = spt.layers.dense(h_x, 1, scope='level_-1')
    return tf.squeeze(h_x, axis=-1)


@add_arg_scope
@spt.global_reuse
def D_kappa(x, y=None):
    # if y is not None:
    #     return D_psi(y) + 0.1 * tf.sqrt(tf.reduce_sum((x - y) ** 2, axis=tf.range(-len(config.x_shape), 0)))
    normalizer_fn = None
    # x = tf.round(256.0 * x / 2 + 127.5)
    # x = (x - 127.5) / 256.0 * 2

    # compute the hidden features
    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=True,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = x
        h_z = spt.layers.dense(h_z, 40)
        h_z = spt.layers.dense(h_z, 40)
        h_z = spt.layers.dense(h_z, 40)
        h_z = spt.layers.dense(h_z, 40)
        h_z = spt.layers.dense(h_z, 40)
        h_z = spt.layers.dense(h_z, 40)
    # sample z ~ q(z|x)
    h_z = spt.layers.dense(h_z, 1, scope='level_-1')
    return tf.squeeze(h_z, axis=-1)


@add_arg_scope
@spt.global_reuse
def p_net(observed=None, n_z=None, beta=1.0, mcmc_iterator=0, log_Z=0.0, initial_z=None,
          mcmc_alpha=config.smallest_step):
    net = spt.BayesianNet(observed=observed)
    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, config.z_dim]),
                        logstd=tf.zeros([1, config.z_dim]))
    normal = normal.batch_ndims_to_value(1)
    xi = tf.get_variable(name='xi', shape=(), initializer=tf.constant_initializer(config.initial_xi),
                         dtype=tf.float32, trainable=True)
    # xi = tf.square(xi)
    xi = tf.nn.sigmoid(xi)  # TODO
    pz = EnergyDistribution(normal, G=G_theta, D=D_psi, log_Z=log_Z, xi=xi, mcmc_iterator=mcmc_iterator,
                            initial_z=initial_z, mcmc_alpha=mcmc_alpha)
    z = net.add('z', pz, n_samples=n_z)
    x_mean = G_theta(z)
    x_mean = tf.clip_by_value(x_mean, 1e-7, 1 - 1e-7)
    logits = tf.log(x_mean) - tf.log1p(-x_mean)
    bernouli = spt.Bernoulli(
        logits=logits
    )
    # bernouli.mean = x_mean
    x = net.add('x', bernouli, group_ndims=3)

    return net


@add_arg_scope
@spt.global_reuse
def p_omega_net(observed=None, n_z=None, beta=1.0, mcmc_iterator=0, log_Z=0.0, initial_z=None):
    net = spt.BayesianNet(observed=observed)
    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, config.z_dim]),
                        logstd=tf.zeros([1, config.z_dim]))
    normal = normal.batch_ndims_to_value(1)
    z = net.add('z', normal, n_samples=n_z)
    x_mean = G_omega(z)
    f_z = net.add('f_z', spt.Normal(mean=x_mean, std=1.0), group_ndims=1)
    x_mean = G_theta(x_mean)
    x = net.add('x', spt.Normal(mean=x_mean, std=1.0), group_ndims=3)
    return net


def get_gradient_penalty(input_origin_x, sample_x, space='x', D=D_psi):
    if space == 'z':
        x = input_origin_x
        x_ = sample_x
        alpha = tf.random_uniform((config.batch_size, 1), minval=0, maxval=1.0)
        x = tf.reshape(x, (-1, config.z_dim))
        x_ = tf.reshape(x_, (-1, config.z_dim))
        x_shape = config.x_shape
        differences = x - x_
        interpolates = x_ + alpha * differences
        interpolates = G_theta(interpolates)
        x = G_theta(x)
        x_ = G_theta(x_)
    elif space == 'x':
        x = input_origin_x
        x_ = sample_x
        alpha = tf.random_uniform(tf.concat([[config.batch_size], [1] * len(config.x_shape)], axis=0), minval=0,
                                  maxval=1.0)
        x = tf.reshape(x, (-1,) + config.x_shape)
        x_ = tf.reshape(x_, (-1,) + config.x_shape)
        x_shape = config.x_shape
        differences = x - x_
        interpolates = x_ + alpha * differences
    else:
        x = input_origin_x
        x_ = sample_x
        alpha = tf.random_uniform((config.batch_size, 1), minval=0, maxval=1.0)
        x = tf.reshape(x, (-1, config.z_dim))
        x_ = tf.reshape(x_, (-1, config.z_dim))
        x_shape = (config.z_dim,)
        differences = x - x_
        interpolates = x_ + alpha * differences

    gradient_penalty = 0.0

    if config.gradient_penalty_algorithm == 'interpolate':
        # print(interpolates)
        D_interpolates = D(interpolates)
        # print(D_interpolates)
        gradient_penalty = tf.square(tf.gradients(D_interpolates, [interpolates])[0])
        gradient_penalty = tf.sqrt(tf.reduce_sum(gradient_penalty, tf.range(-len(x_shape), 0)))
        gradient_penalty = gradient_penalty ** (config.gradient_penalty_index / 2.0)
        gradient_penalty = tf.reduce_mean(gradient_penalty) * config.gradient_penalty_weight

    if config.gradient_penalty_algorithm == 'interpolate-gp':
        print(interpolates)
        D_interpolates = D(interpolates)
        print(D_interpolates)
        gradient_penalty = tf.square(tf.gradients(D_interpolates, [interpolates])[0])
        gradient_penalty = tf.sqrt(tf.reduce_sum(gradient_penalty, tf.range(-len(x_shape), 0))) - 1.0
        gradient_penalty = gradient_penalty ** (config.gradient_penalty_index / 2.0)
        gradient_penalty = tf.reduce_mean(gradient_penalty) * config.gradient_penalty_weight

    if config.gradient_penalty_algorithm == 'both':
        # Sample from fake and real
        energy_real = D(x)
        energy_fake = D(x_)
        gradient_penalty_real = tf.square(tf.gradients(energy_real, [x.tensor if hasattr(x, 'tensor') else x])[0])
        gradient_penalty_real = tf.reduce_sum(gradient_penalty_real, tf.range(-len(x_shape), 0))
        gradient_penalty_real = tf.pow(gradient_penalty_real, config.gradient_penalty_index / 2.0)

        gradient_penalty_fake = tf.square(
            tf.gradients(energy_fake, [x_.tensor if hasattr(x_, 'tensor') else x_])[0])
        gradient_penalty_fake = tf.reduce_sum(gradient_penalty_fake, tf.range(-len(x_shape), 0))
        gradient_penalty_fake = tf.pow(gradient_penalty_fake, config.gradient_penalty_index / 2.0)

        gradient_penalty = (tf.reduce_mean(gradient_penalty_fake) + tf.reduce_mean(gradient_penalty_real)) \
                           * config.gradient_penalty_weight / 2.0
    return gradient_penalty


def get_all_loss(q_net, p_net, pn_omega, pn_theta, warm=1.0, input_origin_x=None):
    with tf.name_scope('adv_prior_loss'):
        gp_omega = get_gradient_penalty(q_net['z'] if config.use_flow else q_net['z'].distribution.mean, pn_omega['f_z'].distribution.mean, D=D_kappa,
                                        space='f_z')
        gp_theta = get_gradient_penalty(input_origin_x, pn_theta['x'].distribution.mean)
        if config.use_dg:
            gp_dg = get_gradient_penalty(q_net['z'], pn_theta['z'], space='z')
        else:
            gp_dg = 0.0

        # VAE_loss = tf.reduce_mean(
        #     -log_px_z - p_net['z'].log_prob() + q_net['z'].log_prob()
        # )
        log_px_z = p_net['x'].log_prob()
        global train_recon
        train_recon = tf.reduce_mean(log_px_z)
        global train_recon_pure_energy
        global train_recon_energy
        global train_kl
        global train_grad_penalty
        another_log_Z = spt.ops.log_mean_exp(
            -p_net['z'].log_prob().energy - q_net['z'].log_prob() + np.log(config.len_train)
        )
        train_log_Z = spt.ops.log_mean_exp(
            tf.stack(
                [p_net['z'].distribution.log_Z, another_log_Z], axis=0
            )
        )

        p_net['z'].distribution.set_log_Z(train_log_Z)
        train_recon_pure_energy = tf.reduce_mean(D_psi(p_net['x'].distribution.mean, p_net['x']))
        train_recon_energy = p_net['z'].log_prob().energy
        train_kl = tf.reduce_mean(
            -p_net['z'].distribution.log_prob(p_net['z'], group_ndims=1, y=p_net['x']).log_energy_prob +
            q_net['z'].log_prob()
        )
        train_grad_penalty = config.pull_back_energy_weight * (gp_theta + gp_dg)  # + gp_omega
        train_kl = tf.maximum(train_kl, 0.0)  # TODO
        VAE_nD_loss = -train_recon + train_kl
        VAE_loss = VAE_nD_loss + train_grad_penalty
        VAE_G_loss = tf.reduce_mean(D_psi(pn_theta['x'].distribution.mean))
        VAE_D_real = tf.reduce_mean(D_psi(input_origin_x))
        VAE_D_loss = -VAE_G_loss + VAE_D_real + train_grad_penalty

        energy_fake = D_kappa(pn_omega['f_z'].distribution.mean)
        energy_real = D_kappa(q_net['z'] if config.use_flow else q_net['z'].distribution.mean)

        adv_D_loss = -tf.reduce_mean(energy_fake) + tf.reduce_mean(
            energy_real) + gp_omega
        adv_G_loss = tf.reduce_mean(energy_fake)
        adv_D_real = tf.reduce_mean(energy_real)
    return VAE_nD_loss, VAE_loss, VAE_D_loss, VAE_G_loss, VAE_D_real, adv_D_loss, adv_G_loss, adv_D_real


class MyIterator(object):
    def __init__(self, iterator):
        self._iterator = iter(iterator)
        self._next = None
        self._has_next = True
        self.next()

    @property
    def has_next(self):
        return self._has_next

    def next(self):
        if not self._has_next:
            raise StopIteration()

        ret = self._next
        try:
            self._next = next(self._iterator)
        except StopIteration:
            self._next = None
            self._has_next = False
        else:
            self._has_next = True
        return ret

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


def limited(iterator, n):
    i = 0
    try:
        while i < n:
            yield next(iterator)
            i += 1
    except StopIteration:
        pass


def get_var(name):
    pfx = name.rsplit('/', 1)
    if len(pfx) == 2:
        vars = tf.global_variables(pfx[0] + '/')
    else:
        vars = tf.global_variables()
    for var in vars:
        if var.name.split(':', 1)[0] == name:
            return var
    raise NameError('Variable {} not exist.'.format(name))


train_recon = None
train_recon_pure_energy = None
train_recon_energy = None
train_kl = None
train_grad_penalty = None


def main():
    # parse the arguments
    arg_parser = ArgumentParser()
    spt.register_config_arguments(config, arg_parser, title='Model options')
    spt.register_config_arguments(spt.settings, arg_parser, prefix='tfsnippet',
                                  title='TFSnippet options')
    arg_parser.parse_args(sys.argv[1:])

    # print the config
    print_with_title('Configurations', pformat(config.to_dict()), after='\n')

    # open the result object and prepare for result directories
    results = MLResults(config.result_dir)
    results.save_config(config)  # save experiment settings for review
    results.make_dirs('plotting/sample', exist_ok=True)
    results.make_dirs('plotting/z_plot', exist_ok=True)
    results.make_dirs('plotting/train.reconstruct', exist_ok=True)
    results.make_dirs('plotting/test.reconstruct', exist_ok=True)
    results.make_dirs('train_summary', exist_ok=True)

    posterior_flow = spt.layers.planar_normalizing_flows(
        config.nf_layers, name='posterior_flow')

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.int32, shape=(None,) + config.x_shape, name='input_x')
    input_origin_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_origin_x')
    warm = tf.placeholder(
        dtype=tf.float32, shape=(), name='warm')
    mcmc_alpha = tf.placeholder(
        dtype=tf.float32, shape=(1,), name='mcmc_alpha')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)
    beta = tf.Variable(initial_value=0.0, dtype=tf.float32, name='beta', trainable=True)

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm], training=True):
        train_pn_theta = p_net(n_z=config.train_n_pz, beta=beta)
        train_pn_omega = p_omega_net(n_z=config.train_n_pz, beta=beta)
        train_log_Z = spt.ops.log_mean_exp(-train_pn_theta['z'].log_prob().energy - train_pn_theta['z'].log_prob())
        train_q_net = q_net(input_origin_x, posterior_flow, n_z=config.train_n_qz)
        train_p_net = p_net(observed={'x': input_origin_x, 'z': train_q_net['z']},
                            n_z=config.train_n_qz, beta=beta, log_Z=train_log_Z)

        VAE_nD_loss, VAE_loss, VAE_D_loss, VAE_G_loss, VAE_D_real, D_loss, G_loss, D_real = get_all_loss(
            train_q_net, train_p_net, train_pn_omega, train_pn_theta, warm, input_origin_x)
        VAE_loss += tf.losses.get_regularization_loss()
        VAE_nD_loss += tf.losses.get_regularization_loss()
        D_loss += tf.losses.get_regularization_loss()
        G_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_origin_x, posterior_flow, n_z=config.test_n_qz)
        # test_pd_net = p_net(n_z=config.test_n_pz // 20, mcmc_iterator=20, beta=beta, log_Z=get_log_Z())
        test_pn_net = p_net(n_z=config.test_n_pz, mcmc_iterator=0, beta=beta, log_Z=get_log_Z())
        test_chain = test_q_net.chain(p_net, observed={'x': tf.to_float(input_x)}, n_z=config.test_n_qz, latent_axis=0,
                                      beta=beta, log_Z=get_log_Z())
        test_mse = tf.reduce_sum(
            (tf.round(test_chain.model['x'].distribution.mean * 255.0) - tf.round(
                tf.to_float(test_chain.model['x']) * 255.0)) ** 2,
            axis=[-1, -2, -3])  # (sample_dim, batch_dim, x_sample_dim)
        test_mse = tf.reduce_min(test_mse, axis=[0])
        test_mse = tf.reduce_mean(tf.reduce_mean(tf.reshape(
            test_mse, (-1, config.test_x_samples,)
        ), axis=-1))
        test_nll = -tf.reduce_mean(
            tf.reshape(
                test_chain.vi.evaluation.is_loglikelihood(), (-1, config.test_x_samples,)
            )
        )
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())
        test_recon = test_chain.model['x'].log_prob()
        p_z = test_chain.model['z'].distribution.log_prob(
            test_chain.model['z'], group_ndims=1, y=test_chain.model['x']
        ).log_energy_prob
        q_z_given_x = test_q_net['z'].log_prob()


        vi = spt.VariationalInference(
            log_joint=test_recon + p_z,
            latent_log_probs=[q_z_given_x],
            axis=0
        )
        test_recon = tf.reduce_mean(test_recon)
        adv_test_nll = -tf.reduce_mean(
            tf.reshape(
                vi.evaluation.is_loglikelihood(), (-1, config.test_x_samples,)
            )
        )
        adv_test_lb = tf.reduce_mean(vi.lower_bound.elbo())

        real_energy = tf.reduce_mean(D_psi(input_origin_x))
        reconstruct_energy = tf.reduce_mean(D_psi(test_chain.model['x'].distribution.mean))
        pd_energy = tf.reduce_mean(
            D_psi(test_pn_net['x'].distribution.mean) * tf.exp(
                test_pn_net['z'].log_prob().log_energy_prob - test_pn_net['z'].log_prob()))
        pn_energy = tf.reduce_mean(D_psi(test_pn_net['x'].distribution.mean))
        log_Z_compute_op = spt.ops.log_mean_exp(
            -test_pn_net['z'].log_prob().energy - test_pn_net['z'].log_prob())

        p_z_energy = test_chain.model['z'].log_prob().energy

        another_log_Z_compute_op = spt.ops.log_mean_exp(
            -p_z_energy - q_z_given_x + np.log(config.len_train)
        )
        kl_adv_and_gaussian = tf.reduce_mean(
            test_pn_net['z'].log_prob() - test_pn_net['z'].log_prob().log_energy_prob
        )
    xi_node = get_var('p_net/xi')
    # derive the optimizer
    with tf.name_scope('optimizing'):
        VAE_params = tf.trainable_variables('q_net') + tf.trainable_variables('G_theta') + tf.trainable_variables(
            'beta') + tf.trainable_variables('posterior_flow') + tf.trainable_variables(
            'p_net/xi') + tf.trainable_variables('D_psi')
        VAE_nD_params = tf.trainable_variables('q_net') + tf.trainable_variables('G_theta') + tf.trainable_variables(
            'beta') + tf.trainable_variables('posterior_flow') + tf.trainable_variables('p_net/xi')
        D_psi_params = tf.trainable_variables('D_psi')
        D_kappa_params = tf.trainable_variables('D_kappa')
        G_params = tf.trainable_variables('G_omega')
        with tf.variable_scope('VAE_optimizer'):
            VAE_optimizer = tf.train.AdamOptimizer(learning_rate)
            VAE_grads = VAE_optimizer.compute_gradients(VAE_loss, VAE_params)
        with tf.variable_scope('VAE_nD_optimizer'):
            VAE_nD_optimizer = tf.train.AdamOptimizer(learning_rate)
            VAE_nD_grads = VAE_nD_optimizer.compute_gradients(VAE_nD_loss, VAE_nD_params)
        with tf.variable_scope('D_psi_optimizer'):
            VAE_D_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            VAE_D_grads = VAE_D_optimizer.compute_gradients(VAE_D_loss, D_psi_params)
        with tf.variable_scope('D_kappa_optimizer'):
            D_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            D_grads = D_optimizer.compute_gradients(D_loss, D_kappa_params)
        with tf.variable_scope('G_optimizer'):
            G_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            G_grads = G_optimizer.compute_gradients(G_loss, G_params)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            VAE_train_op = VAE_optimizer.apply_gradients(VAE_grads)
            VAE_nD_train_op = VAE_optimizer.apply_gradients(VAE_nD_grads)
            VAE_D_train_op = VAE_D_optimizer.apply_gradients(VAE_D_grads)
            G_train_op = G_optimizer.apply_gradients(G_grads)
            D_train_op = D_optimizer.apply_gradients(D_grads)

        # derive the plotting function
        with tf.name_scope('plotting'):
            sample_n_z = config.sample_n_z
            gan_net = p_omega_net(n_z=sample_n_z, beta=beta)
            gan_plots = tf.reshape(gan_net['x'].distribution.mean, (-1,) + config.x_shape)
            gan_z = gan_net['f_z'].distribution.mean
            initial_z = tf.placeholder(
                dtype=tf.float32, shape=(sample_n_z, 1, config.z_dim), name='initial_z')
            gan_plots = 255.0 * gan_plots
            plot_net = p_net(n_z=sample_n_z, mcmc_iterator=20, beta=beta, initial_z=initial_z, mcmc_alpha=mcmc_alpha)
            plot_origin_net = p_net(n_z=sample_n_z, mcmc_iterator=0, beta=beta, initial_z=initial_z)
            plot_history_e_z = plot_net['z'].history_e_z
            plot_history_z = plot_net['z'].history_z
            plot_history_pure_e_z = plot_net['z'].history_pure_e_z
            plot_history_ratio = plot_net['z'].history_ratio
            x_plots = 255.0 * tf.reshape(
                plot_net['x'].distribution.mean, (-1,) + config.x_shape)
            x_origin_plots = 255.0 * tf.reshape(
                plot_origin_net['x'].distribution.mean, (-1,) + config.x_shape)
            reconstruct_q_net = q_net(input_origin_x, posterior_flow)
            reconstruct_z = reconstruct_q_net['z']
            reconstruct_plots = 255.0 * tf.reshape(
                p_net(observed={'z': reconstruct_z}, beta=beta)['x'].distribution.mean,
                (-1,) + config.x_shape
            )
            plot_reconstruct_energy = D_psi(reconstruct_plots)
            gan_z_pure_energy = plot_net['z'].distribution.log_prob(gan_z).pure_energy
            gan_z_energy = plot_net['z'].distribution.log_prob(gan_z).energy
            gan_plots = tf.clip_by_value(gan_plots, 0, 255)
            x_plots = tf.clip_by_value(x_plots, 0, 255)
            x_origin_plots = tf.clip_by_value(x_origin_plots, 0, 255)
            reconstruct_plots = tf.clip_by_value(reconstruct_plots, 0, 255)

        def plot_samples(loop, extra_index=None):
            if extra_index is None:
                extra_index = loop.epoch
            with loop.timeit('plot_time'):
                # plot reconstructs
                for [x] in reconstruct_test_flow:
                    x_samples = bernouli_sampler.sample(x)
                    images = np.zeros((300,) + config.x_shape, dtype=np.uint8)
                    images[::3, ...] = np.round(255.0 * x)
                    images[1::3, ...] = np.round(255.0 * x_samples)
                    batch_reconstruct_plots, batch_reconstruct_z = session.run(
                        [reconstruct_plots, reconstruct_z], feed_dict={input_x: x_samples, input_origin_x: x})
                    images[2::3, ...] = np.round(batch_reconstruct_plots)
                    # print(np.mean(batch_reconstruct_z ** 2, axis=-1))
                    save_images_collection(
                        images=images,
                        filename='plotting/test.reconstruct/{}.png'.format(extra_index),
                        grid_size=(20, 15),
                        results=results,
                    )
                    break

                for [x] in reconstruct_train_flow:
                    x_samples = bernouli_sampler.sample(x)
                    images = np.zeros((300,) + config.x_shape, dtype=np.uint8)
                    images[::3, ...] = np.round(255.0 * x)
                    images[1::3, ...] = np.round(255.0 * x_samples)
                    batch_reconstruct_plots, batch_reconstruct_z = session.run(
                        [reconstruct_plots, reconstruct_z], feed_dict={input_x: x_samples, input_origin_x: x})
                    images[2::3, ...] = np.round(batch_reconstruct_plots)
                    # print(np.mean(batch_reconstruct_z ** 2, axis=-1))
                    save_images_collection(
                        images=images,
                        filename='plotting/train.reconstruct/{}.png'.format(extra_index),
                        grid_size=(20, 15),
                        results=results,
                    )
                    break

                if loop.epoch > config.warm_up_start:
                    # plot samples
                    with loop.timeit('gan_sample_time'):
                        gan_images, batch_z, batch_z_energy, batch_z_pure_energy = session.run(
                            [gan_plots, gan_z, gan_z_energy, gan_z_pure_energy])

                    try:
                        save_images_collection(
                            images=np.round(gan_images),
                            filename='plotting/sample/gan-{}.png'.format(extra_index),
                            grid_size=(10, 10),
                            results=results,
                        )
                    except Exception as e:
                        print(e)

                    mala_images = None
                    if loop.epoch >= config.max_epoch:

                        step_length = config.smallest_step
                        with loop.timeit('mala_sample_time'):
                            for i in range(0, 101):
                                [images, batch_history_e_z, batch_history_z, batch_history_pure_e_z,
                                 batch_history_ratio] = session.run(
                                    [x_plots, plot_history_e_z, plot_history_z, plot_history_pure_e_z,
                                     plot_history_ratio],
                                    feed_dict={
                                        initial_z: batch_z,
                                        mcmc_alpha: np.asarray([step_length])
                                    })
                                batch_z = batch_history_z[-1]

                                if i % 100 == 0:
                                    print(np.mean(batch_history_pure_e_z[-1]), np.mean(batch_history_e_z[-1]))
                                    try:
                                        save_images_collection(
                                            images=np.round(images),
                                            filename='plotting/sample/{}-MALA-{}.png'.format(extra_index, i),
                                            grid_size=(10, 10),
                                            results=results,
                                        )
                                    except Exception as e:
                                        print(e)

                        mala_images = images

                    return mala_images

    # prepare for training and testing data
    (_x_train, _y_train), (_x_test, _y_test) = \
        load_omniglot(x_shape=config.x_shape)
    # train_flow = bernoulli_flow(
    #     x_train, config.batch_size, shuffle=True, skip_incomplete=True)
    x_train = _x_train / 255.0
    x_test = _x_test / 255.0
    bernouli_sampler = BernoulliSampler()
    train_flow = spt.DataFlow.arrays([x_train, x_train], config.batch_size, shuffle=True, skip_incomplete=True)
    train_flow = train_flow.map(lambda x, y: [bernouli_sampler.sample(x), y])
    Z_compute_flow = spt.DataFlow.arrays([x_train, x_train], config.test_batch_size, shuffle=True, skip_incomplete=True)
    Z_compute_flow = Z_compute_flow.map(lambda x, y: [bernouli_sampler.sample(x), y])
    reconstruct_train_flow = spt.DataFlow.arrays(
        [x_train], 100, shuffle=True, skip_incomplete=False)
    reconstruct_test_flow = spt.DataFlow.arrays(
        [x_test], 100, shuffle=True, skip_incomplete=False)

    test_flow = spt.DataFlow.arrays(
        [x_test, x_test],
        config.test_batch_size
    )
    test_flow = test_flow.map(lambda x, y: [bernouli_sampler.sample(x), y])
    # mapped_test_flow = test_flow.to_arrays_flow(config.test_batch_size).map(bernouli_sampler)
    # gathered_flow = spt.DataFlow.gather([test_flow, mapped_test_flow])

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        # initialize the network
        # for [x, origin_x] in train_flow:
        #     print('Network initialized, first-batch loss is {:.6g}.\n'.
        #           format(session.run(init_loss, feed_dict={input_x: x, input_origin_x: origin_x})))
        #     break

        # if config.z_dim == 512:
        #     restore_checkpoint = '/mnt/mfs/mlstorage-experiments/cwx17/48/19/6f3b6c3ef49ded8ba2d5/checkpoint/checkpoint/checkpoint.dat-390000'
        # elif config.z_dim == 1024:
        #     restore_checkpoint = '/mnt/mfs/mlstorage-experiments/cwx17/cd/19/6f9d69b5d1931e67e2d5/checkpoint/checkpoint/checkpoint.dat-390000'
        # elif config.z_dim == 2048:
        #     restore_checkpoint = '/mnt/mfs/mlstorage-experiments/cwx17/4d/19/6f9d69b5d19398c8c2d5/checkpoint/checkpoint/checkpoint.dat-390000'
        # elif config.z_dim == 3072:
        #     restore_checkpoint = '/mnt/mfs/mlstorage-experiments/cwx17/5d/19/6f9d69b5d1936fb2d2d5/checkpoint/checkpoint/checkpoint.dat-390000'
        # else:
        restore_checkpoint = None  # '/mnt/mfs/mlstorage-experiments/cwx17/2c/fb/d4e63c432be9319e0cd5/checkpoint/checkpoint/checkpoint.dat-312000'

        # train the network
        with spt.TrainLoop(tf.trainable_variables(),
                           var_groups=['q_net', 'p_net', 'posterior_flow', 'G_theta', 'D_psi', 'G_omega', 'D_kappa'],
                           max_epoch=config.max_epoch,
                           max_step=config.max_step,
                           summary_dir=(results.system_path('train_summary')
                                        if config.write_summary else None),
                           summary_graph=tf.get_default_graph(),
                           early_stopping=False,
                           checkpoint_dir=results.system_path('checkpoint'),
                           checkpoint_epoch_freq=100,
                           restore_checkpoint=restore_checkpoint
                           ) as loop:

            evaluator = spt.Evaluator(
                loop,
                metrics={'test_nll': test_nll, 'test_lb': test_lb,
                         'adv_test_nll': adv_test_nll, 'adv_test_lb': adv_test_lb,
                         'reconstruct_energy': reconstruct_energy,
                         'real_energy': real_energy,
                         'pd_energy': pd_energy, 'pn_energy': pn_energy,
                         'test_recon': test_recon, 'kl_adv_and_gaussian': kl_adv_and_gaussian, 'test_mse': test_mse},
                inputs=[input_x, input_origin_x],
                data_flow=test_flow,
                time_metric_name='test_time'
            )

            loop.print_training_summary()
            spt.utils.ensure_variables_initialized()

            epoch_iterator = loop.iter_epochs()

            n_critical = config.n_critical
            # adversarial training
            for epoch in epoch_iterator:

                if epoch <= config.warm_up_start:
                    step_iterator = MyIterator(train_flow)
                    while step_iterator.has_next:
                        for step, [x, origin_x] in loop.iter_steps(limited(step_iterator, n_critical)):
                            # vae training
                            [_, batch_VAE_loss, beta_value, xi_value, batch_train_recon, batch_train_recon_energy,
                             batch_train_recon_pure_energy, batch_VAE_D_real, batch_VAE_G_loss, batch_train_kl,
                             batch_train_grad_penalty] = session.run(
                                [VAE_nD_train_op, VAE_loss, beta, xi_node, train_recon, train_recon_energy,
                                 train_recon_pure_energy, VAE_D_real, VAE_G_loss,
                                 train_kl, train_grad_penalty],
                                feed_dict={
                                    input_x: x,
                                    input_origin_x: origin_x,
                                    warm: 1.0  # min(1.0, 1.0 * epoch / config.warm_up_epoch)
                                })
                            loop.collect_metrics(batch_VAE_loss=batch_VAE_loss)
                            loop.collect_metrics(xi=xi_value)
                            loop.collect_metrics(beta=beta_value)
                            loop.collect_metrics(train_recon=batch_train_recon)
                            loop.collect_metrics(train_recon_pure_energy=batch_train_recon_pure_energy)
                            loop.collect_metrics(train_recon_energy=batch_train_recon_energy)
                            loop.collect_metrics(VAE_D_real=batch_VAE_D_real)
                            loop.collect_metrics(VAE_G_loss=batch_VAE_G_loss)
                            loop.collect_metrics(train_kl=batch_train_kl)
                            loop.collect_metrics(train_grad_penalty=batch_train_grad_penalty)

                            _ = session.run(VAE_D_train_op, feed_dict={
                                input_x: x,
                                input_origin_x: origin_x,
                                warm: 1.0  # min(1.0, 1.0 * epoch / config.warm_up_epoch)
                            })
                else:
                    step_iterator = MyIterator(train_flow)
                    while step_iterator.has_next:
                        for step, [x, origin_x] in loop.iter_steps(limited(step_iterator, n_critical)):
                            # discriminator training
                            [_, batch_D_loss, batch_D_real] = session.run(
                                [D_train_op, D_loss, D_real], feed_dict={
                                    input_x: x,
                                    input_origin_x: origin_x,
                                })
                            loop.collect_metrics(D_loss=batch_D_loss)
                            loop.collect_metrics(D_real=batch_D_real)

                        # generator training x
                        [_, batch_G_loss] = session.run(
                            [G_train_op, G_loss], feed_dict={
                            })
                        loop.collect_metrics(G_loss=batch_G_loss)

                if epoch in config.lr_anneal_epoch_freq:
                    learning_rate.anneal()

                if epoch == config.warm_up_start:
                    learning_rate.set(config.initial_lr)

                if epoch % config.plot_epoch_freq == 0:
                    plot_samples(loop)

                if epoch % config.test_epoch_freq == 0:
                    with loop.timeit('compute_Z_time'):
                        # log_Z_list = []
                        # for i in range(config.log_Z_times):
                        #     log_Z_list.append(session.run(log_Z_compute_op))
                        # from scipy.misc import logsumexp
                        # log_Z = logsumexp(np.asarray(log_Z_list)) - np.log(len(log_Z_list))
                        # print('log_Z_list:{}'.format(log_Z_list))
                        # print('log_Z:{}'.format(log_Z))

                        log_Z_list = []
                        for [batch_x, batch_origin_x] in Z_compute_flow:
                            log_Z_list.append(session.run(another_log_Z_compute_op, feed_dict={
                                input_x: batch_x,
                                input_origin_x: batch_origin_x
                            }))
                        from scipy.misc import logsumexp
                        another_log_Z = logsumexp(np.asarray(log_Z_list)) - np.log(len(log_Z_list))
                        # print('log_Z_list:{}'.format(log_Z_list))
                        print('another_log_Z:{}'.format(another_log_Z))
                        # final_log_Z = logsumexp(np.asarray([log_Z, another_log_Z])) - np.log(2)
                        final_log_Z = another_log_Z  # TODO
                        get_log_Z().set(final_log_Z)

                    with loop.timeit('eval_time'):
                        evaluator.run()

                if epoch == config.max_epoch:
                    dataset_img = np.tile(_x_train, (1, 1, 1, 3))
                    mala_img = []
                    for i in range(config.fid_samples // config.sample_n_z):
                        mala_images = plot_samples(loop, 10000 + i)
                        mala_img.append(mala_images)
                        print('{}-th sample finished...'.format(i))

                    mala_img = np.concatenate(mala_img, axis=0).astype('uint8')
                    mala_img = np.asarray(mala_img)
                    mala_img = np.tile(mala_img, (1, 1, 1, 3))
                    FID = get_fid(mala_img, dataset_img)
                    IS_mean, IS_std = get_inception_score(mala_img)
                    loop.collect_metrics(FID=FID)
                    loop.collect_metrics(IS=IS_mean)

                loop.collect_metrics(lr=learning_rate.get())
                loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()