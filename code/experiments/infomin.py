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
from code.experiments.energy_distribution import EnergyDistribution
from tfsnippet.examples.utils import (MLResults,
                                      save_images_collection,
                                      bernoulli_as_pixel,
                                      bernoulli_flow,
                                      print_with_title)
import numpy as np
from scipy.misc import logsumexp


class ExpConfig(spt.Config):
    # data options
    channels_last = True

    # model parameters
    z_dim = 40
    act_norm = False
    weight_norm = False
    l2_reg = 0.0001
    kernel_size = 3
    shortcut_kernel_size = 1

    # training parameters
    result_dir = None
    write_summary = True
    max_epoch = 5100
    pretrain_epoch = 300
    episode_epoch = 4800
    adv_train_epoch = 2400

    max_step = None
    batch_size = 128
    initial_lr = 0.001
    lr_anneal_factor = 0.1
    lr_anneal_epoch_freq = 800
    lr_anneal_step_freq = None

    gradient_penalty_weight = 0.1
    kl_balance_weight = 1.0

    n_critical = 5
    Z_compute_epochs = 10
    # evaluation parameters
    train_n_w = 1000
    train_n_z = 1
    test_n_z = 100
    test_batch_size = 64
    test_epoch_freq = 100
    plot_epoch_freq = 10

    epsilon = -20
    Z_batch_limit = 10

    @property
    def x_shape(self):
        return (28, 28, 1) if self.channels_last else (1, 28, 28)


config = ExpConfig()


@add_arg_scope
@spt.global_reuse
def q_net(x, observed=None, n_z=None, is_training=False, is_initializing=False):
    net = spt.BayesianNet(observed=observed)

    normalizer_fn = None if not config.act_norm else functools.partial(
        spt.layers.act_norm,
        scope='act_norm',
        axis=-1 if config.channels_last else -3,
        initializing=is_initializing,
        value_ndims=3,
    )

    # compute the hidden features
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg),
                   channels_last=config.channels_last):
        h_x = tf.to_float(x)
        h_x = spt.layers.conv2d(h_x, 16, scope='level_0')  # output: (16, 28, 28)
        h_x = spt.layers.conv2d(h_x, 32, strides=2, scope='level_1')  # output: (32, 14, 14)
        h_x = spt.layers.conv2d(h_x, 32, scope='level_2')  # output: (32, 14, 14)
        h_x = spt.layers.conv2d(h_x, 64, strides=2, scope='level_3')  # output: (64, 7, 7)
        h_x = spt.layers.conv2d(h_x, 64, scope='level_4')  # output: (64, 7, 7)

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
def p_net(observed=None, n_z=None, gaussian_prior=False, mcmc_on_z=False,
          mcmc_on_w=False, is_training=False,
          is_initializing=False):
    net = spt.BayesianNet(observed=observed)

    normalizer_fn = None if not config.act_norm else functools.partial(
        spt.layers.act_norm,
        scope='act_norm',
        axis=-1 if config.channels_last else -3,
        initializing=is_initializing,
        value_ndims=3,
    )

    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, config.z_dim]),
                        logstd=tf.zeros([1, config.z_dim]))
    normal = normal.batch_ndims_to_value(1)
    pz = EnergyDistribution(normal, G=G_phi, U=U_psi, log_Z=get_log_Z(), mcmc_on_z=mcmc_on_w, mcmc_on_x=mcmc_on_z)
    if gaussian_prior:
        z = net.add('z', normal, n_samples=n_z)
    else:
        z = net.add('z', pz, n_samples=n_z)

    # compute the hidden features
    with arg_scope([spt.layers.resnet_deconv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg),
                   channels_last=config.channels_last):
        h_z = spt.layers.dense(z, 64 * 7 * 7, scope='level_0')
        h_z = spt.ops.reshape_tail(
            h_z,
            ndims=1,
            shape=(7, 7, 64) if config.channels_last else (64, 7, 7)
        )
        h_z = spt.layers.conv2d(h_z, 64, scope='level_1')  # output: (64, 7, 7)
        h_z = spt.layers.conv2d(h_z, 32, strides=2, scope='level_2')  # output: (32, 14, 14)
        h_z = spt.layers.conv2d(h_z, 32, scope='level_3')  # output: (32, 14, 14)
        h_z = spt.layers.conv2d(h_z, 16, strides=2, scope='level_4')  # output: (16, 28, 28)

    # sample x ~ p(x|z)
    x_logits = spt.layers.conv2d(
        h_z, 1, (1, 1), padding='same', scope='feature_map_to_pixel',
        channels_last=config.channels_last
    )  # output: (1, 28, 28)
    x = net.add('x', spt.Bernoulli(logits=x_logits), group_ndims=3)
    return net


@add_arg_scope
@spt.global_reuse
def G_phi(w, is_training=False, is_initializing=False):
    if config.act_norm:
        normalizer_fn = functools.partial(
            spt.layers.act_norm, initializing=is_initializing)
    else:
        normalizer_fn = None

    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=config.weight_norm,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_w = tf.to_float(w)
        h_w = spt.layers.dense(h_w, 100, scope='level_0')
        h_w = spt.layers.dense(h_w, 100, scope='level_1')

    h_w = spt.layers.dense(h_w, config.z_dim, scope='level_2')
    return h_w


@add_arg_scope
@spt.global_reuse
def U_psi(z, is_training=False, is_initializing=False):
    if config.act_norm:
        normalizer_fn = functools.partial(
            spt.layers.act_norm, initializing=is_initializing)
    else:
        normalizer_fn = None

    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=config.weight_norm,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_u = tf.to_float(z)
        h_u = spt.layers.dense(h_u, 100, scope='level_0')
        h_u = spt.layers.dense(h_u, 100, scope='level_1')
    h_u = spt.layers.dense(h_u, 1, scope='level_2')
    return tf.squeeze(h_u, axis=-1)


@add_arg_scope
@spt.global_reuse
def T_omega(z, w, is_training=False, is_initializing=False):
    if config.act_norm:
        normalizer_fn = functools.partial(
            spt.layers.act_norm, initializing=is_initializing)
    else:
        normalizer_fn = None

    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=config.weight_norm,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_t = tf.concat([tf.to_float(z), tf.to_float(w)], axis=-1)
        h_t = spt.layers.dense(h_t, 100, scope='level_0')
        h_t = spt.layers.dense(h_t, 100, scope='level_1')
    h_t = spt.layers.dense(h_t, 1, scope='level_2')
    return tf.squeeze(h_t, axis=-1)


@add_arg_scope
@spt.global_reuse
def R_gamma(z, x, is_training=False, is_initializing=False):
    if config.act_norm:
        normalizer_fn = functools.partial(
            spt.layers.act_norm, initializing=is_initializing)
    else:
        normalizer_fn = None

    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=config.weight_norm,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_r = tf.concat([tf.to_float(z), tf.to_float(x)], axis=-1)
        h_r = spt.layers.dense(h_r, 500, scope='level_0')
        h_r = spt.layers.dense(h_r, 500, scope='level_1')
    h_r = spt.layers.dense(h_r, 1, scope='level_2')
    return tf.squeeze(h_r, axis=-1)


debug_energy_z = None
debug_energy_Gw = None
debug_qz_mean = None
debug_qz_var = None


def adv_prior_loss(q_net, pz_net, pw_net):
    with tf.name_scope('adv_prior_loss'):
        z = q_net['z']
        Gw = pw_net['z']

        z_mean, z_variance = tf.nn.moments(z, axes=[0, 1])
        log_px_z = pz_net['x'].log_prob(name='log_px_z')

        global debug_qz_mean
        debug_qz_mean = z_mean
        global debug_qz_var
        debug_qz_var = z_variance

        energy_z = pz_net['z'].log_prob(name='log_pz').energy
        energy_Gw = pw_net['z'].log_prob(name='log_pGw').energy
        gradient_penalty = tf.reduce_sum(
            tf.square(tf.gradients(energy_z, [z.tensor])[0]), axis=-1
        )
        adv_psi_loss = -tf.reduce_mean(energy_Gw) + tf.reduce_mean(
            energy_z + config.gradient_penalty_weight * gradient_penalty)
        global debug_energy_z
        debug_energy_z = tf.reduce_mean(energy_z)
        global debug_energy_Gw
        debug_energy_Gw = tf.reduce_mean(energy_Gw)

        # I(G(w), w)
        logits_T = T_omega(Gw, Gw.z)
        logits_T_ = T_omega(Gw, Gw.z_)
        concat_logits = tf.concat([logits_T, logits_T_], axis=0)
        concat_label = tf.concat([tf.ones_like(logits_T), tf.zeros_like(logits_T_)], axis=0)
        adv_omega_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=concat_logits, labels=concat_label
        ))
        adv_phi_loss = adv_omega_loss + tf.reduce_mean(energy_Gw)

        # I(x, z)
        logits_R = R_gamma(pz_net['x'], z)
        logits_R_ = R_gamma(tf.random_shuffle(pz_net['x']), z)
        concat_logits = tf.concat([logits_R, logits_R_], axis=0)
        concat_label = tf.concat([tf.ones_like(logits_R), tf.zeros_like(logits_R_)], axis=0)
        adv_gamma_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=concat_logits, labels=concat_label
        ))

        adv_theta_loss = -adv_gamma_loss + tf.reduce_mean(
            -log_px_z
        ) + tf.reduce_mean(tf.square(z_mean)) + tf.reduce_mean(tf.square(z_variance - 1.0))
    return adv_psi_loss, adv_theta_loss, adv_phi_loss, adv_omega_loss, adv_gamma_loss


@spt.utils.add_name_arg_doc
def sgvb_loss(p_net, q_net, beta, metrics_dict, prefix='train_', name=None):
    with tf.name_scope(name, default_name='sgvb_loss'):
        log_px_z = tf.reduce_mean(p_net['x'].log_prob(name='log_px_z'))
        log_pz = tf.reduce_mean(p_net['z'].log_prob(name='log_pz'))
        log_qz_x = tf.reduce_mean(q_net['z'].log_prob(name='log_qz_x'))

        metrics_dict[prefix + 'recons'] = log_px_z
        metrics_dict[prefix + 'log_pz'] = log_pz
        metrics_dict[prefix + 'log_qz_x'] = log_qz_x

        return -(log_px_z + beta * (log_pz - log_qz_x))


def compute_partition_function(train_flow, input_x, q_net, pz_net, pw_net):
    session = spt.utils.get_default_session_or_error()
    log_Z = []
    log_Z_var = []
    for __ in range(config.Z_compute_epochs):
        qz_x_mean = []
        qz_x_std = []
        qz_x = []
        z_energy = []
        for _, [batch_x] in zip(range(config.Z_batch_limit), train_flow):
            batch_qz_x_mean, batch_qz_x_std, batch_qz_x, batch_z_energy = session.run(
                [q_net['z'].distribution.mean, q_net['z'].distribution.std, q_net['z'], pz_net['z'].log_prob().energy],
                feed_dict={input_x: batch_x})
            qz_x_mean.append(batch_qz_x_mean)
            qz_x_std.append(batch_qz_x_std)
            qz_x.append(batch_qz_x)
            z_energy.append(batch_z_energy)
        qz_x_mean = np.concatenate(qz_x_mean, axis=0)  # [x_size, z_dim]
        qz_x_std = np.concatenate(qz_x_std, axis=0)  # [x_size, z_dim]
        qz_x = np.concatenate(qz_x, axis=1)  # [z_samples, x_size, z_dim]
        z_energy = np.concatenate(z_energy, axis=1)
        # [z_samples, x_size, z_dim]

        qz_x = np.expand_dims(qz_x, axis=2)
        # [z_samples, x_size, 1, z_dim]
        print(qz_x_mean.shape, qz_x_std.shape, qz_x.shape, z_energy.shape)

        log_qz_x_ = np.sum(
            -np.square(qz_x - qz_x_mean) / 2.0 / np.square(qz_x_std) - np.log(qz_x_std) - 0.5 * np.log(2.0 * np.pi),
            axis=-1)
        log_qz = logsumexp(log_qz_x_, axis=2)
        # [z_samples, x_size, z_dim]
        log_Z.append(logsumexp(-z_energy - log_qz))
        # log_Z_var.append(
        #     (np.exp(logsumexp((-z_energy - log_qz) * 2.0)) - Z[-1]) / (z_energy.shape[0] * z_energy.shape[1])
        # )
    print(log_Z)
    log_Z = np.mean(np.asarray(log_Z))
    # log_Z_var = np.mean(np.asarray(log_Z_var))

    print('log_Z=%f±%f', log_Z, np.sqrt(log_Z_var))
    get_log_Z().set(log_Z)


@contextmanager
def copy_gradients(gradients):
    gradients = list(gradients)

    with tf.variable_scope('copy_gradients'):
        ret = []
        copied_grads = []
        control_ops = []

        for grad, var in gradients:
            copied_grad = tf.get_variable(
                var.name.rsplit(':', 1)[0].replace('/', '_'),
                dtype=grad.dtype,
                shape=grad.get_shape()
            )
            copied_grads.append(copied_grad)
            grad = spt.utils.maybe_check_numerics(grad, grad.name)
            control_ops.append(tf.assign(copied_grad, grad))

        with tf.control_dependencies(control_ops):
            for grad, (_, var) in zip(copied_grads, gradients):
                ret.append((tf.identity(grad), var))
            yield ret


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
    results.make_dirs('train_summary', exist_ok=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.int32, shape=(None,) + config.x_shape, name='input_x')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)
    beta = tf.placeholder(dtype=tf.float32, shape=(), name='beta')

    # derive the loss for initializing
    with tf.name_scope('initialization'), \
         arg_scope([p_net, q_net], is_initializing=True), \
         spt.utils.scoped_set_config(spt.settings, auto_histogram=False):
        init_q_net = q_net(input_x, n_z=config.train_n_z)
        init_pw_net = p_net(observed={'x': input_x}, n_z=config.train_n_w)
        init_pz_net = p_net(observed={'x': input_x, 'z': init_q_net['z']}, n_z=config.train_n_z)
        init_loss = sum(adv_prior_loss(init_q_net, init_pz_net, init_pw_net))

    # derive the loss and lower-bound for training
    with tf.name_scope('pretraining'), \
         arg_scope([p_net, q_net], is_training=True):
        pretrain_q_net = q_net(input_x)
        pretrain_pz_net = p_net(observed={'x': input_x, 'z': pretrain_q_net['z']},
                                gaussian_prior=True)
        pretrain_theta_loss = sgvb_loss(pretrain_pz_net, pretrain_q_net, beta, {})
        pretrain_theta_loss += tf.losses.get_regularization_loss()

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([p_net, q_net], is_training=True):
        train_q_net = q_net(input_x, n_z=config.train_n_z)
        train_pz_net = p_net(observed={'x': input_x, 'z': train_q_net['z']}, n_z=config.train_n_z)
        train_pw_net = p_net(observed={'x': input_x}, n_z=config.train_n_w)

        adv_psi_loss, adv_theta_loss, adv_phi_loss, adv_omega_loss, adv_gamma_loss = adv_prior_loss(
            train_q_net, train_pz_net, train_pw_net)
        adv_psi_loss += tf.losses.get_regularization_loss()
        adv_theta_loss += tf.losses.get_regularization_loss()
        adv_phi_loss += tf.losses.get_regularization_loss()
        adv_omega_loss += tf.losses.get_regularization_loss()
        adv_gamma_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, n_z=config.test_n_z)
        test_pz_net = p_net(observed={'x': input_x, 'z': test_q_net['z']}, n_z=config.test_n_z)
        test_pw_net = p_net(observed={'x': input_x}, n_z=config.test_n_z)
        test_chain = test_q_net.chain(p_net, observed={'x': input_x}, n_z=config.test_n_z, latent_axis=0)
        test_nll = -tf.reduce_mean(test_chain.vi.evaluation.is_loglikelihood())
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())

    # derive the optimizer
    with tf.name_scope('optimizing'):
        phi_params = tf.trainable_variables('G_phi')
        psi_params = tf.trainable_variables('U_psi')
        theta_params = tf.trainable_variables('p_net') + tf.trainable_variables('q_net')
        omega_params = tf.trainable_variables('T_omega')
        gamma_params = tf.trainable_variables('R_gamma')
        print("========phi=========")
        print(phi_params)
        print("========theta=========")
        print(theta_params)
        print("========omega=========")
        print(omega_params)
        phi_optimizer = tf.train.AdamOptimizer(learning_rate)
        psi_optimizer = tf.train.AdamOptimizer(learning_rate)
        theta_optimizer = tf.train.AdamOptimizer(learning_rate)
        omega_optimizer = tf.train.AdamOptimizer(learning_rate)
        gamma_optimizer = tf.train.AdamOptimizer(learning_rate)
        theta_pretrain_optimizer = tf.train.AdamOptimizer(learning_rate)

        theta_grads = theta_optimizer.compute_gradients(adv_theta_loss, theta_params)
        theta_grads = [(spt.utils.maybe_check_numerics(grad, grad.name), var) for grad, var in theta_grads]
        psi_grads = psi_optimizer.compute_gradients(adv_psi_loss, psi_params)
        psi_grads = [(spt.utils.maybe_check_numerics(grad, grad.name), var) for grad, var in psi_grads]
        theta_pretrain_grads = theta_optimizer.compute_gradients(pretrain_theta_loss, theta_params)
        theta_pretrain_grads = [
            (spt.utils.maybe_check_numerics(grad, grad.name), var) for grad, var in theta_pretrain_grads]
        gamma_grads = theta_optimizer.compute_gradients(adv_gamma_loss, gamma_params)
        gamma_grads = [
            (spt.utils.maybe_check_numerics(grad, grad.name), var) for grad, var in gamma_grads]
        # omega_grads = omega_optimizer.compute_gradients(adv_omega_loss, omega_params)
        # phi_grads = phi_optimizer.compute_gradients(adv_theta_loss, phi_params)

        with copy_gradients(phi_optimizer.compute_gradients(adv_phi_loss, phi_params)) as phi_grads:
            with copy_gradients(omega_optimizer.compute_gradients(adv_omega_loss, omega_params)) as omega_grads:
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    phi_train_op = phi_optimizer.apply_gradients(phi_grads)
                    omega_train_op = omega_optimizer.apply_gradients(omega_grads)

        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            theta_train_op = theta_optimizer.apply_gradients(theta_grads)
            psi_train_op = psi_optimizer.apply_gradients(psi_grads)
            theta_pretrain_op = theta_pretrain_optimizer.apply_gradients(theta_pretrain_grads)
            gamma_train_op = gamma_optimizer.apply_gradients(gamma_grads)

            # phi_train_op = theta_optimizer.apply_gradients(phi_grads)
            # omega_train_op = omega_optimizer.apply_gradients(omega_grads)

    # derive the plotting function
    with tf.name_scope('plotting'):
        x_plots = tf.reshape(
            bernoulli_as_pixel(p_net(n_z=100)['x']), (-1,) + config.x_shape)
        z_plots = tf.reshape(
            p_net(n_z=1000)['z'], (-1, config.z_dim)
        )
        reconstruct_q_net = q_net(input_x)
        reconstruct_z = reconstruct_q_net['z']
        reconstruct_plots = tf.reshape(
            bernoulli_as_pixel(
                p_net(observed={'z': reconstruct_z})['x']),
            (-1,) + config.x_shape
        )

    def plot_samples(loop):
        with loop.timeit('plot_time'):
            # plot samples
            images, z_points = session.run([x_plots, z_plots])
            pyplot.scatter(z_points[:, 0], z_points[:, 1], s=5)
            pyplot.savefig(results.system_path('plotting/z_plot/{}.pdf'.format(loop.epoch)))
            pyplot.close()
            # print(images)
            save_images_collection(
                images=images,
                filename='plotting/sample/{}.png'.format(loop.epoch),
                grid_size=(10, 10),
                results=results,
                channels_last=config.channels_last,
            )

            # plot reconstructs
            for [x] in reconstruct_train_flow:
                x_samples = reconstruct_sampler.sample(x)
                images = np.zeros((100,) + config.x_shape, dtype=np.uint8)
                images[::2, ...] = x.astype(np.uint8)
                images[1::2, ...] = session.run(reconstruct_plots, feed_dict={input_x: x_samples})
                save_images_collection(
                    images=images,
                    filename='plotting/train.reconstruct/{}.png'.format(loop.epoch),
                    grid_size=(10, 10),
                    results=results,
                    channels_last=config.channels_last,
                )
                break

    # prepare for training and testing data
    (x_train, y_train), (x_test, y_test) = \
        spt.datasets.load_mnist(x_shape=config.x_shape)
    train_flow = bernoulli_flow(
        x_train, config.batch_size, shuffle=True, skip_incomplete=True)
    Z_train_flow = bernoulli_flow(
        x_train, config.batch_size, shuffle=True, skip_incomplete=True)
    reconstruct_train_flow = spt.DataFlow.arrays(
        [x_train], 50, shuffle=True, skip_incomplete=False)
    reconstruct_sampler = spt.preprocessing.BernoulliSampler()
    test_flow = bernoulli_flow(
        x_test, config.test_batch_size, sample_now=True)

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
                           max_epoch=config.max_epoch,
                           max_step=config.max_step,
                           summary_dir=(results.system_path('train_summary')
                                        if config.write_summary else None),
                           summary_graph=tf.get_default_graph(),
                           early_stopping=False,
                           checkpoint_dir=results.system_path('checkpoint'),
                           checkpoint_epoch_freq=100) as loop:

            evaluator = spt.Evaluator(
                loop,
                metrics={'test_nll': test_nll, 'test_lb': test_lb},
                inputs=[input_x],
                data_flow=test_flow,
                time_metric_name='test_time'
            )
            evaluator.events.on(
                spt.EventKeys.AFTER_EXECUTION,
                lambda e: results.update_metrics(evaluator.last_metrics_dict)
            )

            loop.print_training_summary()
            spt.utils.ensure_variables_initialized()

            epoch_iterator = loop.iter_epochs()
            # pre-training
            for epoch in limited(epoch_iterator, config.pretrain_epoch - loop.epoch):
                step_iterator = MyIterator(loop.iter_steps(train_flow))
                for step, [x] in step_iterator:
                    _, batch_theta_loss = session.run(
                        [theta_pretrain_op, pretrain_theta_loss], feed_dict={
                            input_x: x,
                            beta: min(1., epoch / 100.)
                        })
                    loop.collect_metrics(pretrain_theta_loss=batch_theta_loss)
                if epoch % config.lr_anneal_epoch_freq == 0:
                    learning_rate.anneal()
                loop.print_logs()

            # adversarial training
            for epoch in epoch_iterator:
                step_iterator = MyIterator(loop.iter_steps(train_flow))
                episode_epoch = (epoch - config.pretrain_epoch - 1) % config.episode_epoch

                if episode_epoch == 0 or episode_epoch == config.adv_train_epoch:
                    learning_rate.set(config.initial_lr)

                if episode_epoch >= config.adv_train_epoch:
                    while step_iterator.has_next:
                        # energy training
                        for step, [x] in limited(step_iterator, config.n_critical):
                            [_, batch_psi_loss, _z, _w] = session.run(
                                [psi_train_op, adv_psi_loss, debug_energy_z, debug_energy_Gw], feed_dict={
                                    input_x: x
                                })
                            loop.collect_metrics(energy_z=_z)
                            loop.collect_metrics(energy_Gw=_w)
                            loop.collect_metrics(adv_psi_loss=batch_psi_loss)

                        # generator training x
                        [_, __, batch_phi_loss, batch_omega_loss, _w] = session.run(
                            [phi_train_op, omega_train_op, adv_phi_loss, adv_omega_loss,
                             debug_energy_Gw])
                        loop.collect_metrics(energy_Gw=_w)
                        loop.collect_metrics(adv_phi_loss=batch_phi_loss)
                        loop.collect_metrics(adv_omega_loss=batch_omega_loss)
                else:
                    while step_iterator.has_next:
                        for step, [x] in step_iterator:
                            # vae training
                            [_, batch_theta_loss, _z, _w] = session.run(
                                [theta_train_op, adv_theta_loss, debug_energy_z, debug_energy_Gw], feed_dict={
                                    input_x: x
                                })
                            loop.collect_metrics(energy_z=_z)
                            loop.collect_metrics(energy_Gw=_w)
                            loop.collect_metrics(adv_theta_loss=batch_theta_loss)

                        for step, [x] in step_iterator:
                            # vae training
                            [_, batch_gamma_loss, _z, _w] = session.run(
                                [theta_train_op, adv_gamma_loss, debug_energy_z, debug_energy_Gw], feed_dict={
                                    input_x: x
                                })
                            loop.collect_metrics(energy_z=_z)
                            loop.collect_metrics(energy_Gw=_w)
                            loop.collect_metrics(adv_theta_loss=batch_gamma_loss)

                if epoch % config.lr_anneal_epoch_freq == 0:
                    learning_rate.anneal()

                if epoch % config.test_epoch_freq == 0:
                    compute_partition_function(Z_train_flow, input_x, test_q_net, test_pz_net, test_pw_net)
                    with loop.timeit('eval_time'):
                        evaluator.run()

                if epoch % config.plot_epoch_freq == 0:
                    plot_samples(loop)

                if epoch == config.max_epoch:
                    compute_partition_function(Z_train_flow, input_x, test_q_net, test_pz_net, test_pw_net)
                    evaluator.run()

                loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
