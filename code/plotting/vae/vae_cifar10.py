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
from code.experiment.utils import get_inception_score, get_fid
from tfsnippet.examples.utils import (MLResults,
                                      save_images_collection,
                                      bernoulli_as_pixel,
                                      bernoulli_flow,
                                      print_with_title)
import numpy as np
from scipy.misc import logsumexp

from tfsnippet.preprocessing import UniformNoiseSampler


class ExpConfig(spt.Config):
    # model parameters
    z_dim = 512
    act_norm = False
    weight_norm = False
    l2_reg = 0.0002
    kernel_size = 3
    shortcut_kernel_size = 1
    batch_norm = True

    # training parameters
    result_dir = None
    write_summary = True
    max_epoch = 1500
    energy_prior_start_epoch = 1500
    beta = 1e-8
    pull_back_energy_weight = 1

    max_step = None
    batch_size = 128
    initial_lr = 0.0002
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 400
    lr_anneal_step_freq = None

    gradient_penalty_weight = 2
    gradient_penalty_index = 6
    kl_balance_weight = 1.0

    n_critical = 1
    # evaluation parameters
    train_n_pz = 128
    train_n_qz = 1
    test_n_pz = 5000
    test_n_qz = 10
    test_batch_size = 64
    test_epoch_freq = 100
    plot_epoch_freq = 10

    test_x_samples = 8
    truncated_sigma = 1.0

    fid_samples = 50000
    sample_n_z = 100

    epsilon = -20

    @property
    def x_shape(self):
        return (32, 32, 3)

    x_shape_multiple = 32 * 32 * 3


config = ExpConfig()


class EnergyDistribution(spt.Distribution):
    """
    A distribution derived from an energy function `D(x)` and a generator
    function `x = G(z)`, where `p(z) = exp(-D(G(z))) / Z`.
    """

    def __init__(self, pz, G, D, log_Z=0., mcmc_iterator=0, mcmc_alpha=0.01, mcmc_algorithm='mala', mcmc_space='z'):
        """
        Construct a new :class:`EnergyDistribution`.

        Args:
            pz (spt.Distribution): The base distribution `p(z)`.
            G: The function `x = G(z)`.
            D: The function `D(x)`.
            Z: The partition factor `Z`.
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
        with tf.name_scope('log_Z', values=[log_Z]):
            self._log_Z = tf.maximum(log_Z, -20)
        self._mcmc_iterator = mcmc_iterator
        self._mcmc_alpha = mcmc_alpha

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
    def log_Z(self):
        return self._log_Z

    def log_prob(self, given, group_ndims=0, name=None):
        given = tf.convert_to_tensor(given)
        with tf.name_scope(name,
                           default_name=spt.utils.get_default_scope_name(
                               'log_prob', self),
                           values=[given]):
            energy = self.D(self.G(given)) * config.pull_back_energy_weight + 0.5 * tf.reduce_sum(tf.square(given),
                                                                                                  axis=-1)
            log_px = self.pz.log_prob(given=given, group_ndims=group_ndims)
            log_px.log_energy_prob = -energy - self.log_Z
            log_px.energy = energy

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
            for i in range(self._mcmc_iterator):
                e_z, grad_e_z, z_prime = self.get_sgld_proposal(z)
                e_z_prime, grad_e_z_prime, _ = self.get_sgld_proposal(z_prime)

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
                # print(ratio.mean().item())
                rnd_u = tf.random.normal(
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
        return t

    def get_sgld_proposal(self, z):
        energy_z = self.D(self.G(z)) * config.pull_back_energy_weight + 0.5 * tf.reduce_sum(tf.square(z), axis=-1)
        grad_energy_z = tf.gradients(energy_z, [z.tensor if hasattr(z, 'tensor') else z])[0]
        grad_energy_z = tf.reshape(grad_energy_z, shape=z.shape)
        eps = tf.random.normal(
            shape=z.shape
        ) * np.sqrt(self._mcmc_alpha * 2)
        z_prime = z - self._mcmc_alpha * grad_energy_z + eps
        return energy_z, grad_energy_z, z_prime


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


@add_arg_scope
def batch_norm(inputs, training=False, scope=None):
    return tf.layers.batch_normalization(inputs, training=training, name=scope)


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
def q_net(x, observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = tf.to_float(x)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (28, 28, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_1')  # output: (28, 28, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_2')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_3')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, strides=2, scope='level_4')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, scope='level_5')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_6')  # output: (7, 7, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, scope='level_7')  # output: (7, 7, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256, strides=2, scope='level_8')  # output: (7, 7, 64)

    # sample z ~ q(z|x)
    h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
    z_mean = spt.layers.dense(h_x, config.z_dim, scope='z_mean', kernel_initializer=tf.zeros_initializer())
    z_logstd = spt.layers.dense(h_x, config.z_dim, scope='z_logstd', kernel_initializer=tf.zeros_initializer())
    z = net.add('z', spt.Normal(mean=z_mean, logstd=spt.ops.maybe_clip_value(z_logstd, min_val=config.epsilon)),
                n_samples=n_z, group_ndims=1)

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
def p_net(observed=None, n_z=None, beta=1.0, mcmc_iterator=0):
    normalizer_fn = None
    net = spt.BayesianNet(observed=observed)
    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, config.z_dim]),
                        std=tf.ones([1, config.z_dim]) * config.truncated_sigma)
    normal = normal.batch_ndims_to_value(1)
    pz = EnergyDistribution(normal, G=G_theta, D=D_psi, log_Z=get_log_Z(), mcmc_iterator=mcmc_iterator)
    z = net.add('z', pz, n_samples=n_z)
    x_mean = G_theta(z)
    x = net.add('x', ExponentialDistribution(
        mean=x_mean,
        beta=beta,
        D=D_psi
    ), group_ndims=3)
    # compute the hidden features
    return net


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
        h_z = spt.layers.dense(z, 256 * config.x_shape[0] // 8 * config.x_shape[1] // 8, scope='level_0',
                               normalizer_fn=None)
        h_z = spt.ops.reshape_tail(
            h_z,
            ndims=1,
            shape=(config.x_shape[0] // 8, config.x_shape[1] // 8, 256)
        )
        h_z = spt.layers.resnet_deconv2d_block(h_z, 256, strides=2, scope='level_1')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, scope='level_2')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_3')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 64, scope='level_4')  # output: (14, 14, 32)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 64, strides=2, scope='level_5')  # output: (14, 14, 32)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32, scope='level_6')  # output: (14, 14, 32)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32, scope='level_7')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16, scope='level_8')  # output: (28, 28, 16)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16, scope='level_9')  # output: (28, 28, 16)
    x_mean = spt.layers.conv2d(
        h_z, config.x_shape[-1], (1, 1), padding='same', scope='feature_map_mean_to_pixel',
        kernel_initializer=tf.zeros_initializer()
    )
    return x_mean


@add_arg_scope
@spt.global_reuse
def D_psi(x):
    normalizer_fn = None

    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg), ):
        h_x = tf.to_float(x)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (28, 28, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_1')  # output: (28, 28, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_2')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_3')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, strides=2, scope='level_4')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, scope='level_5')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_6')  # output: (7, 7, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, scope='level_7')  # output: (7, 7, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256, strides=2, scope='level_8')  # output: (7, 7, 64)

        h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
        h_x = spt.layers.dense(h_x, 64, scope='level_5')
    # sample z ~ q(z|x)
    h_x = spt.layers.dense(h_x, 1, scope='level_6')
    return tf.squeeze(h_x, axis=-1)


def get_all_loss(q_net, p_net):
    with tf.name_scope('adv_prior_loss'):
        log_px_z = p_net['x'].log_prob()
        VAE_loss = tf.reduce_mean(
            -log_px_z - p_net['z'].log_prob() + q_net['z'].log_prob()
        )
        global debug_variable
        debug_variable = tf.reduce_mean(
            tf.sqrt(tf.reduce_sum((p_net['x'] - p_net['x'].distribution.mean) ** 2, [2, 3, 4])))

    return VAE_loss


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


debug_variable = None


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

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_x')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)
    beta = tf.Variable(initial_value=0.1, dtype=tf.float32, name='beta', trainable=True)
    beta = tf.clip_by_value(beta, config.beta, 1.0)

    # derive the loss for initializing
    with tf.name_scope('initialization'), \
         arg_scope([spt.layers.act_norm], initializing=True), \
         spt.utils.scoped_set_config(spt.settings, auto_histogram=False):
        init_q_net = q_net(input_x, n_z=config.train_n_qz)
        init_p_net = p_net(observed={'x': input_x, 'z': init_q_net['z']}, n_z=config.train_n_qz, beta=beta)
        init_loss = get_all_loss(init_q_net, init_p_net)

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm], training=True):
        train_q_net = q_net(input_x, n_z=config.train_n_qz)
        train_p_net = p_net(observed={'x': input_x, 'z': train_q_net['z']}, n_z=config.train_n_qz, beta=beta)

        VAE_loss = get_all_loss(train_q_net, train_p_net)
        VAE_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, n_z=config.test_n_qz)
        test_chain = test_q_net.chain(p_net, observed={'x': input_x}, n_z=config.test_n_qz, latent_axis=0,
                                      beta=beta)
        test_nll = -tf.reduce_mean(
            spt.ops.log_mean_exp(
                tf.reshape(
                    test_chain.vi.evaluation.is_loglikelihood(), (-1, config.test_x_samples,)
                ), axis=-1)
        ) + config.x_shape_multiple * np.log(128.0)

        test_recon = tf.reduce_mean(
            test_chain.model['x'].log_prob()
        )
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())
        test_mse = tf.reduce_sum(
            (tf.round(test_chain.model['x'].distribution.mean * 128 + 127.5) - tf.round(
                test_chain.model['x'] * 128 + 127.5)) ** 2, axis=[-1, -2, -3])  # (sample_dim, batch_dim, x_sample_dim)
        test_mse = tf.reduce_min(test_mse, axis=[0])
        test_mse = tf.reduce_mean(tf.reduce_mean(tf.reshape(
            test_mse, (-1, config.test_x_samples,)
        ), axis=-1))

    # derive the optimizer
    with tf.name_scope('optimizing'):
        VAE_params = tf.trainable_variables('q_net') + tf.trainable_variables('G_theta') + tf.trainable_variables(
            'beta')
        print("========VAE_params=========")
        print(VAE_params)
        with tf.variable_scope('VAE_optimizer'):
            VAE_optimizer = tf.train.AdamOptimizer(learning_rate)
        VAE_grads = VAE_optimizer.compute_gradients(VAE_loss, VAE_params)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            VAE_train_op = VAE_optimizer.apply_gradients(VAE_grads)

    # derive the plotting function
    with tf.name_scope('plotting'):
        x_plots = 256.0 * tf.reshape(
            p_net(n_z=100, mcmc_iterator=0, beta=beta)['x'].distribution.mean, (-1,) + config.x_shape) / 2 + 127.5
        reconstruct_q_net = q_net(input_x)
        reconstruct_z = reconstruct_q_net['z']
        reconstruct_plots = 256.0 * tf.reshape(
            p_net(observed={'z': reconstruct_z}, beta=beta)['x'],
            (-1,) + config.x_shape
        ) / 2 + 127.5
        x_plots = tf.clip_by_value(x_plots, 0, 255)
        reconstruct_plots = tf.clip_by_value(reconstruct_plots, 0, 255)

    def plot_samples(loop):
        with loop.timeit('plot_time'):
            # plot samples
            images = session.run(x_plots)
            # pyplot.scatter(z_points[:, 0], z_points[:, 1], s=5)
            # pyplot.savefig(results.system_path('plotting/z_plot/{}.pdf'.format(loop.epoch)))
            # pyplot.close()
            # print(images)
            try:
                print(np.max(images), np.min(images))
                images = np.round(images)
                save_images_collection(
                    images=images,
                    filename='plotting/sample/{}.png'.format(loop.epoch),
                    grid_size=(10, 10),
                    results=results,
                )

                # plot reconstructs
                for [x] in reconstruct_train_flow:
                    x_samples = uniform_sampler.sample(x)
                    images = np.zeros((150,) + config.x_shape, dtype=np.uint8)
                    images[::3, ...] = np.round(256.0 * x / 2 + 127.5)
                    images[1::3, ...] = np.round(256.0 * x_samples / 2 + 127.5)
                    images[2::3, ...] = np.round(session.run(
                        reconstruct_plots, feed_dict={input_x: x}))
                    save_images_collection(
                        images=images,
                        filename='plotting/train.reconstruct/{}.png'.format(loop.epoch),
                        grid_size=(10, 15),
                        results=results,
                    )
                    break

                # plot reconstructs
                for [x] in reconstruct_test_flow:
                    x_samples = uniform_sampler.sample(x)
                    images = np.zeros((150,) + config.x_shape, dtype=np.uint8)
                    images[::3, ...] = np.round(256.0 * x / 2 + 127.5)
                    images[1::3, ...] = np.round(256.0 * x_samples / 2 + 127.5)
                    images[2::3, ...] = np.round(session.run(
                        reconstruct_plots, feed_dict={input_x: x}))
                    save_images_collection(
                        images=images,
                        filename='plotting/test.reconstruct/{}.png'.format(loop.epoch),
                        grid_size=(10, 15),
                        results=results,
                    )
                    break
            except Exception as e:
                print(e)

    # prepare for training and testing data
    (_x_train, _y_train), (_x_test, _y_test) = \
        spt.datasets.load_cifar10(x_shape=config.x_shape)
    # train_flow = bernoulli_flow(
    #     x_train, config.batch_size, shuffle=True, skip_incomplete=True)
    x_train = (_x_train - 127.5) / 256.0 * 2
    x_test = (_x_test - 127.5) / 256.0 * 2
    uniform_sampler = UniformNoiseSampler(-1.0 / 256.0, 1.0 / 256.0, dtype=np.float)
    train_flow = spt.DataFlow.arrays([x_train], config.batch_size, shuffle=True, skip_incomplete=True)
    train_flow = train_flow.map(uniform_sampler)
    reconstruct_train_flow = spt.DataFlow.arrays(
        [x_train], 50, shuffle=True, skip_incomplete=False)
    reconstruct_test_flow = spt.DataFlow.arrays(
        [x_test], 50, shuffle=True, skip_incomplete=False)
    test_flow = spt.DataFlow.arrays(
        [np.repeat(x_test, config.test_x_samples, axis=0)], config.test_batch_size)
    test_flow = test_flow.map(uniform_sampler)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        # initialize the network
        for [x] in train_flow:
            print('Network initialized, first-batch loss is {:.6g}.\n'.
                  format(session.run(init_loss, feed_dict={input_x: x})))
            break

        # train the network
        with spt.TrainLoop(tf.trainable_variables(),
                           var_groups=['q_net', 'p_net'],
                           max_epoch=config.max_epoch + 1,
                           max_step=config.max_step,
                           summary_dir=(results.system_path('train_summary')
                                        if config.write_summary else None),
                           summary_graph=tf.get_default_graph(),
                           early_stopping=False,
                           checkpoint_dir=results.system_path('checkpoint'),
                           checkpoint_epoch_freq=100,
                           restore_checkpoint="/mnt/mfs/mlstorage-experiments/cwx17/83/19/6f3b6c3ef49d6d6c81d5/checkpoint/checkpoint/checkpoint.dat-585000"
                           ) as loop:

            evaluator = spt.Evaluator(
                loop,
                metrics={'test_nll': test_nll, 'test_lb': test_lb, 'test_recon': test_recon, 'test_mse': test_mse},
                inputs=[input_x],
                data_flow=test_flow,
                time_metric_name='test_time'
            )

            loop.print_training_summary()
            spt.utils.ensure_variables_initialized()
            epoch_iterator = loop.iter_epochs()
            for epoch in epoch_iterator:
                plot_samples(loop)
                dataset_img = _x_train
                mala_img = []
                for i in range(config.fid_samples // config.sample_n_z):
                    mala_images = session.run(x_plots)
                    mala_img.append(mala_images)
                    print('{}-th sample finished...'.format(i))

                mala_img = np.concatenate(mala_img, axis=0).astype('uint8')
                mala_img = np.asarray(mala_img)
                np.savez('sample_store', mala_img=mala_img)

                FID = get_fid(mala_img, dataset_img)
                IS_mean, IS_std = get_inception_score(mala_img)
                loop.collect_metrics(FID=FID)
                loop.collect_metrics(IS=IS_mean)

                # ori_img = np.concatenate(ori_img, axis=0).astype('uint8')
                # ori_img = np.asarray(ori_img)
                # FID = get_fid_google(ori_img, dataset_img)
                # IS_mean, IS_std = get_inception_score(ori_img)
                # loop.collect_metrics(FID_ori=FID)
                # loop.collect_metrics(IS_ori=IS_mean)

                loop.collect_metrics(lr=learning_rate.get())
                loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
