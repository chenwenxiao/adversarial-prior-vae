# -*- coding: utf-8 -*-
import functools
import sys
from argparse import ArgumentParser

import tensorflow as tf
from pprint import pformat
from tensorflow.contrib.framework import arg_scope, add_arg_scope

import tfsnippet as spt
from code.experiment.utils import get_fid, get_inception_score
from tfsnippet.examples.utils import (MLResults,
                                      save_images_collection,
                                      bernoulli_as_pixel,
                                      bernoulli_flow,
                                      print_with_title)
from tfsnippet.preprocessing import BernoulliSampler
import numpy as np


class ExpConfig(spt.Config):
    # data options
    channels_last = True

    # model parameters
    z_dim = 40
    act_norm = True
    l2_reg = 0.0001
    kernel_size = 3
    shortcut_kernel_size = 1

    # training parameters
    result_dir = None
    write_summary = False
    max_epoch = 300
    max_step = None
    batch_size = 128
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 100
    lr_anneal_step_freq = None
    use_q_z_given_e = False
    use_origin_x_as_observe = False

    # evaluation parameters
    test_n_z = 10
    test_batch_size = 64

    fid_samples = 50000
    sample_n_z = 100
    truncated_sigma = 1.0

    @property
    def x_shape(self):
        return (28, 28, 1) if self.channels_last else (1, 28, 28)


config = ExpConfig()


def _bernoulli_mean(self):
    if not hasattr(self, '_mean'):
        self._mean = tf.sigmoid(self.logits)
    return self._mean


spt.Bernoulli.mean = property(_bernoulli_mean)


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
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, strides=2)  # output: (32, 14, 14)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32)  # output: (32, 14, 14)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, strides=2)  # output: (64, 7, 7)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64)  # output: (64, 7, 7)

    # sample z ~ q(z|x)
    h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
    z_mean = spt.layers.dense(h_x, config.z_dim, name='z_mean')
    z_logstd = spt.layers.dense(h_x, config.z_dim, name='z_logstd')
    z = net.add('z', spt.Normal(mean=z_mean, logstd=z_logstd), n_samples=n_z,
                group_ndims=1)

    return net


@spt.global_reuse
@add_arg_scope
def p_net(observed=None, n_z=None, is_training=False, is_initializing=False):
    net = spt.BayesianNet(observed=observed)

    normalizer_fn = None if not config.act_norm else functools.partial(
        spt.layers.act_norm,
        axis=-1 if config.channels_last else -3,
        initializing=is_initializing,
        value_ndims=3,
    )

    # sample z ~ p(z)
    z = net.add('z', spt.Normal(mean=tf.zeros([1, config.z_dim]),
                                std=tf.ones([1, config.z_dim]) * config.truncated_sigma),
                group_ndims=1, n_samples=n_z)

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
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32, strides=2)  # output: (32, 14, 14)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32)  # output: (32, 14, 14)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16, strides=2)  # output: (16, 28, 28)

    # sample x ~ p(x|z)
    x_logits = spt.layers.conv2d(
        h_z, 1, (1, 1), padding='same', name='feature_map_to_pixel',
        channels_last=config.channels_last
    )  # output: (1, 28, 28)
    x = net.add('x', spt.Bernoulli(logits=x_logits, dtype=tf.float32), group_ndims=3)

    return net


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
    input_origin_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_origin_x')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)
    input_x = tf.to_float(input_x)

    # derive the loss for initializing
    with tf.name_scope('initialization'), \
         arg_scope([p_net, q_net], is_initializing=True), \
         spt.utils.scoped_set_config(spt.settings, auto_histogram=False):
        init_q_net = q_net(input_origin_x if config.use_q_z_given_e else input_x)
        init_chain = init_q_net.chain(p_net,
                                      observed={'x': input_origin_x if config.use_origin_x_as_observe else input_x})
        init_loss = tf.reduce_mean(init_chain.vi.training.sgvb())

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([p_net, q_net], is_training=True):
        train_q_net = q_net(input_origin_x if config.use_q_z_given_e else input_x)
        train_chain = train_q_net.chain(p_net,
                                        observed={'x': input_origin_x if config.use_origin_x_as_observe else input_x})
        train_loss = (
            tf.reduce_mean(train_chain.vi.training.sgvb()) +
            tf.losses.get_regularization_loss()
        )

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_origin_x if config.use_q_z_given_e else input_x, n_z=config.test_n_z)
        test_chain = test_q_net.chain(
            p_net, latent_axis=0, observed={'x': tf.to_float(input_x)})
        test_nll = -tf.reduce_mean(test_chain.vi.evaluation.is_loglikelihood())
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())
        test_mse = tf.reduce_sum(
            (tf.round(test_chain.model['x'].distribution.mean * 128 + 127.5) - tf.round(
                input_origin_x * 128 + 127.5)) ** 2, axis=[-1, -2, -3])  # (sample_dim, batch_dim)
        test_mse = tf.reduce_min(test_mse, axis=[0])
        test_mse = tf.reduce_mean(test_mse)

    # derive the optimizer
    with tf.name_scope('optimizing'):
        params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradients(train_loss, params)
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.apply_gradients(grads)

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
    (_x_train, _y_train), (_x_test, _y_test) = \
        spt.datasets.load_mnist(x_shape=config.x_shape)
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

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        # initialize the network
        for [x, ox] in train_flow:
            print('Network initialized, first-batch loss is {:.6g}.\n'.
                  format(session.run(init_loss, feed_dict={input_x: x, input_origin_x: ox})))
            break

        # train the network
        with spt.TrainLoop(params,
                           var_groups=['q_net', 'p_net'],
                           max_epoch=config.max_epoch + 1,
                           max_step=config.max_step,
                           summary_dir=(results.system_path('train_summary')
                                        if config.write_summary else None),
                           summary_graph=tf.get_default_graph(),
                           checkpoint_dir=results.system_path('checkpoint'),
                           checkpoint_epoch_freq=100,
                           early_stopping=False,
                           restore_checkpoint="/mnt/mfs/mlstorage-experiments/cwx17/10/1c/d4e63c432be97afba7e5/checkpoint/checkpoint/checkpoint.dat-140400"
                           ) as loop:

            loop.print_training_summary()
            spt.utils.ensure_variables_initialized()

            epoch_iterator = loop.iter_epochs()
            for epoch in epoch_iterator:
                dataset_img = np.tile(_x_train, (1, 1, 1, 3))
                mala_img = []
                for i in range(config.fid_samples // config.sample_n_z):
                    mala_images = session.run(x_plots)
                    mala_img.append(mala_images)
                    print('{}-th sample finished...'.format(i))

                mala_img = np.concatenate(mala_img, axis=0).astype('uint8')
                mala_img = np.asarray(mala_img)
                mala_img = np.tile(mala_img, (1, 1, 1, 3))
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
