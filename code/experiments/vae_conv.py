# -*- coding: utf-8 -*-
import functools
import sys
from argparse import ArgumentParser
from contextlib import contextmanager

import six
import tensorflow as tf
from pprint import pformat
from tensorflow.contrib.framework import arg_scope, add_arg_scope

import tfsnippet as spt
from tfsnippet import resolve_feed_dict, merge_feed_dict
from tfsnippet.examples.auto_encoders.energy_distribution import \
    EnergyDistribution
from tfsnippet.examples.utils import (MLResults,
                                      save_images_collection,
                                      bernoulli_as_pixel,
                                      bernoulli_flow,
                                      print_with_title)
import numpy as np
from scipy.misc import logsumexp

from tfsnippet.utils import is_tensor_object


class ExpConfig(spt.Config):
    # data options
    channels_last = True

    # model parameters
    z_dim = 40
    act_norm = True
    weight_norm = True
    l2_reg = 0.0001
    kernel_size = 3
    shortcut_kernel_size = 1

    # training parameters
    result_dir = None
    write_summary = False
    max_epoch = 100
    max_step = None
    batch_size = 128
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 300
    lr_anneal_step_freq = None

    gradient_penalty_weight = 0.1
    kl_balance_weight = 1.0

    n_critical = 5
    # evaluation parameters
    test_n_z = 500
    test_batch_size = 64
    test_epoch_freq = 100

    @property
    def x_shape(self):
        return (28, 28, 1) if self.channels_last else (1, 28, 28)


config = ExpConfig()


@spt.global_reuse
@add_arg_scope
def q_net(x, observed=None, n_z=None, is_training=False, is_initializing=False):
    net = spt.BayesianNet(observed=observed)

    normalizer_fn = None if not config.act_norm else functools.partial(
        spt.layers.act_norm,
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
        h_x = spt.layers.resnet_conv2d_block(h_x, 16)  # output: (16, 28, 28)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32,
                                             strides=2)  # output: (32, 14, 14)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32)  # output: (32, 14, 14)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64,
                                             strides=2)  # output: (64, 7, 7)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64)  # output: (64, 7, 7)

    # sample z ~ q(z|x)
    h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
    z_mean = spt.layers.dense(h_x, config.z_dim, name='z_mean')
    z_logstd = spt.layers.dense(h_x, config.z_dim, name='z_logstd')
    z = net.add('z', spt.Normal(mean=z_mean, logstd=z_logstd), n_samples=n_z,
                group_ndims=1)

    return net


@spt.global_reuse
def get_Z():
    return spt.model_variable('Z', dtype=tf.float32, initializer=1.,
                              trainable=False)


@spt.global_reuse
@add_arg_scope
def p_net(observed=None, n_z=None, mcmc_on_z=False,
          mcmc_on_w=False, is_training=False,
          is_initializing=False):
    net = spt.BayesianNet(observed=observed)

    normalizer_fn = None if not config.act_norm else functools.partial(
        spt.layers.act_norm,
        axis=-1 if config.channels_last else -3,
        initializing=is_initializing,
        value_ndims=3,
    )

    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, config.z_dim]),
                        logstd=tf.zeros([1, config.z_dim]))
    pz = EnergyDistribution(
        normal.batch_ndims_to_value(1), G=G_phi, U=U_theta, Z=get_Z(),
        mcmc_on_z=mcmc_on_w, mcmc_on_x=mcmc_on_z)
    z = net.add('z', pz, n_samples=n_z)

    # compute the hidden features
    with arg_scope([spt.layers.resnet_deconv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg),
                   channels_last=config.channels_last):
        h_z = spt.layers.dense(z, 64 * 7 * 7)
        h_z = spt.ops.reshape_tail(
            h_z,
            ndims=1,
            shape=(7, 7, 64) if config.channels_last else (64, 7, 7)
        )
        h_z = spt.layers.resnet_deconv2d_block(h_z, 64)  # output: (64, 7, 7)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32,
                                               strides=2)  # output: (32, 14, 14)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32)  # output: (32, 14, 14)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16,
                                               strides=2)  # output: (16, 28, 28)

    # sample x ~ p(x|z)
    x_logits = spt.layers.conv2d(
        h_z, 1, (1, 1), padding='same', name='feature_map_to_pixel',
        channels_last=config.channels_last
    )  # output: (1, 28, 28)
    x = net.add('x', spt.Bernoulli(logits=x_logits), group_ndims=3)
    return net


@spt.global_reuse
@add_arg_scope
def G_phi(w, is_training=False, is_initializing=False):
    normalizer_fn = functools.partial(
        spt.layers.act_norm, initializing=is_initializing)

    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=config.weight_norm,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_w = tf.to_float(w)
        h_w = spt.layers.dense(h_w, 500)
        h_w = spt.layers.dense(h_w, 500)

    h_w = spt.layers.dense(h_w, config.z_dim)
    return h_w


@spt.global_reuse
@add_arg_scope
def U_theta(z, is_training=False, is_initializing=False):
    normalizer_fn = functools.partial(
        spt.layers.act_norm, initializing=is_initializing)

    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=config.weight_norm,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_u = tf.to_float(z)
        h_u = spt.layers.dense(h_u, 500)
        h_u = spt.layers.dense(h_u, 500)
    h_u = spt.layers.dense(h_u, 1)
    return tf.squeeze(h_u, axis=-1)


@spt.global_reuse
@add_arg_scope
def T_omega(z, w, is_training=False, is_initializing=False):
    normalizer_fn = functools.partial(
        spt.layers.act_norm, initializing=is_initializing)

    with arg_scope([spt.layers.dense],
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=config.weight_norm,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_t = tf.concat([tf.to_float(z), tf.to_float(w)], axis=-1)
        h_t = spt.layers.dense(h_t, 500)
        h_t = spt.layers.dense(h_t, 500)
    h_t = spt.layers.dense(h_t, 1, activation_fn=tf.nn.sigmoid)
    return tf.squeeze(h_t, axis=-1)


def adv_prior_loss(q_net, pz_net, pw_net):
    with tf.name_scope('adv_prior_loss'):
        z = q_net['z']
        Gw = pw_net['z']
        log_px_z = pz_net['x'].log_prob()

        energy_z = pz_net['z'].log_prob().energy
        energy_Gw = pw_net['z'].log_prob().energy
        gradient_penalty = tf.reduce_sum(
            tf.square(tf.gradients(energy_z, [z])[0]), axis=-1
        )
        adv_theta_loss = -tf.reduce_mean(energy_Gw) + tf.reduce_mean(
            -log_px_z + energy_z +
            config.gradient_penalty_weight * gradient_penalty
        )
        adv_omega_loss = tf.reduce_mean(
            tf.log(T_omega(Gw, Gw.z)) - tf.log(1 - T_omega(Gw, Gw.z_))
        )
        adv_phi_loss = adv_omega_loss + energy_Gw + config.kl_balance_weight * (z.log_prob() - log_px_z + energy_z)
    return adv_theta_loss, adv_phi_loss, adv_omega_loss


def compute_partition_function(train_flow, q_net, pz_net, pw_net):
    with spt.utils.create_session().as_default() as session:
        qz_x_mean = []
        qz_x_std = []
        qz_x = []
        z_energy = []
        for [batch_x] in train_flow:
            batch_qz_x_mean, batch_qz_x_std, batch_qz_x, batch_z_energy = session.run(
                [q_net['z'].distribution.mean, q_net['z'].distribution.std, q_net['z'], pz_net['z'].log_prob().energy],
                feed_dict={'input_x': batch_x})
            qz_x_mean.append(batch_qz_x_mean)
            qz_x_std.append(batch_qz_x_std)
            qz_x.append(batch_qz_x)
            z_energy.append(batch_z_energy)
        qz_x_mean = np.concatenate(qz_x_mean, axis=0)  # [x_size, z_dim]
        qz_x_std = np.concatenate(qz_x_std, axis=0)  # [x_size, z_dim]
        qz_x = np.concatenate(qz_x, axis=1)  # [z_samples, x_size, z_dim]
        z_energy = np.concatenate(z_energy, axis=1)
        # [z_samples, x_size, z_dim]
        qz_x_mean = np.expand_dims(qz_x_mean, axis=2)
        # [z_samples, x_size, 1, z_dim]

        log_qz_x_ = np.sum(
            -np.square(qz_x - qz_x_mean) / 2 / np.square(qz_x_std) - np.log(qz_x_std) - 0.5 * np.log(2 * np.pi),
            axis=-1)
        log_qz = logsumexp(log_qz_x_, axis=2)
        # [z_samples, x_size, z_dim]
        Z = np.mean(-z_energy - log_qz)
        print('Z=%f', Z)
    tf.assign(get_Z, Z)


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
            control_ops.append(tf.assign(copied_grad, var))

        with tf.control_dependencies(control_ops):
            for grad, (_, var) in zip(copied_grads, gradients):
                ret.append((tf.identity(grad), var))
            yield ret


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
    results.make_dirs('plotting', exist_ok=True)
    results.make_dirs('train_summary', exist_ok=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.int32, shape=(None,) + config.x_shape, name='input_x')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # derive the loss for initializing
    with tf.name_scope('initialization'), \
         arg_scope([p_net, q_net], is_initializing=True), \
         spt.utils.scoped_set_config(spt.settings, auto_histogram=False):
        init_q_net = q_net(input_x)
        init_chain = init_q_net.chain(p_net, observed={'x': input_x})
        init_loss = tf.reduce_mean(init_chain.vi.training.sgvb())

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([p_net, q_net], is_training=True):
        train_q_net = q_net(input_x)
        train_pz_net = p_net(observed={'x': input_x, 'z': train_q_net['z']})
        train_pw_net = p_net(observed={'x': input_x})

        adv_theta_loss, adv_phi_loss, adv_omega_loss = adv_prior_loss(train_q_net, train_pz_net, train_pw_net)
        adv_theta_loss += tf.losses.get_regularization_loss()
        adv_phi_loss += tf.losses.get_regularization_loss()
        adv_omega_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, n_z=config.test_n_z)
        test_p_net = p_net(observed={'x': input_x, 'z': test_q_net['z']},
                           n_z=config.test_n_z)
        test_chain = test_q_net.chain(test_p_net)
        test_nll = -tf.reduce_mean(test_chain.vi.evaluation.is_loglikelihood())
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())

    # derive the optimizer
    with tf.name_scope('optimizing'):
        phi_params = tf.trainable_variables('G_phi') + tf.trainable_variables('q_net')
        theta_params = tf.trainable_variables('U_theta') + tf.trainable_variables('p_net')
        omega_params = tf.trainable_variables('T_omega')
        phi_optimizer = tf.train.AdamOptimizer(learning_rate)
        theta_optimizer = tf.train.AdamOptimizer(learning_rate)
        omega_optimizer = tf.train.AdamOptimizer(learning_rate)

        theta_grads = theta_optimizer.compute_gradients(adv_theta_loss, theta_params)

        with copy_gradients(phi_optimizer.compute_gradients(adv_theta_loss, phi_params)) as phi_grads:
            with copy_gradients(omega_optimizer.compute_gradients(adv_omega_loss, omega_params)) as omega_grads:
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    phi_train_op = theta_optimizer.apply_gradients(phi_grads)
                    with tf.control_dependencies([phi_train_op]):
                        omega_train_op = omega_optimizer.apply_gradients(omega_grads)

        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            theta_train_op = theta_optimizer.apply_gradients(theta_grads)

    # derive the plotting function
    with tf.name_scope('plotting'):
        x_plots = tf.reshape(
            bernoulli_as_pixel(p_net(n_z=100)['x']), (-1,) + config.x_shape)

    def plot_samples(loop):
        with loop.timeit('plot_time'):
            images = session.run(x_plots)
            save_images_collection(
                images=images,
                filename='plotting/{}.png'.format(loop.epoch),
                grid_size=(10, 10),
                results=results,
                channels_last=config.channels_last,
            )

    # prepare for training and testing data
    (x_train, y_train), (x_test, y_test) = \
        spt.datasets.load_mnist(x_shape=config.x_shape)
    train_flow = bernoulli_flow(
        x_train, config.batch_size, shuffle=True, skip_incomplete=True)
    Z_train_flow = bernoulli_flow(
        x_train, config.batch_size, shuffle=False, skip_incomplete=False)
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
                           early_stopping=False) as loop:

            def do_evaluate():
                compute_partition_function(Z_train_flow, train_q_net, train_pz_net, train_pw_net)
                evaluator.run()
                plot_samples(loop)

            for epoch in loop.iter_epochs():
                step_iterator = loop.iter_steps(train_flow)

                try:
                    while True:
                        # discriminator training
                        for i in range(config.n_critical):
                            step, [x] = next(step_iterator)
                            session.run(theta_train_op, feed_dict={
                                input_x: x
                            })

                        # generator training
                        step, [x] = next(step_iterator)
                        session.run([phi_train_op, omega_train_op], feed_dict={
                            input_x: x
                        })
                except StopIteration:
                    pass

                if epoch % config.lr_anneal_epoch_freq == 0:
                    learning_rate.anneal()
                loop.print_logs()

                if epoch % config.test_epoch_freq:
                    do_evaluate()

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

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
