# -*- coding: utf-8 -*-
import functools
import sys
from argparse import ArgumentParser

import tensorflow as tf
from pprint import pformat
from tensorflow.contrib.framework import arg_scope, add_arg_scope

import tfsnippet as spt
from tfsnippet import VariationalInference
from tfsnippet.examples.utils import (MLResults,
                                      save_images_collection,
                                      bernoulli_as_pixel,
                                      bernoulli_flow,
                                      print_with_title)
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
    max_epoch = 2400
    warm_up_epoch = 300
    converge_epoch_begin = 1200
    converge_epoch_length = 300
    max_step = None
    batch_size = 128
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 300
    lr_anneal_step_freq = None
    train_n_x = 10

    # evaluation parameters
    test_n_z = 100
    test_n_x = 40
    test_batch_size = 1
    test_epoch_freq = 100
    plot_epoch_freq = 10

    final_test_n_z = 500
    final_test_n_x = 100

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
    with arg_scope([spt.layers.conv2d],
                   kernel_size=config.kernel_size,
                   # shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg),
                   channels_last=config.channels_last):
        h_x = tf.to_float(x)
        h_x = spt.layers.conv2d(h_x, 16)  # output: (16, 28, 28)
        h_x = spt.layers.conv2d(h_x, 32, strides=2)  # output: (32, 14, 14)
        h_x = spt.layers.conv2d(h_x, 32)  # output: (32, 14, 14)
        h_x = spt.layers.conv2d(h_x, 64, strides=2)  # output: (64, 7, 7)
        h_x = spt.layers.conv2d(h_x, 64)  # output: (64, 7, 7)

    # sample z ~ q(z|x)
    h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
    z_mean = spt.layers.dense(h_x, config.z_dim, name='z_mean')
    z_logstd = spt.layers.dense(h_x, config.z_dim, name='z_logstd')
    z = net.add('z', spt.Normal(mean=z_mean, logstd=z_logstd), n_samples=n_z,
                group_ndims=1)

    return net


@spt.global_reuse
@add_arg_scope
def p_net(observed=None, n_z=None, n_x=None, is_training=False, is_initializing=False):
    net = spt.BayesianNet(observed=observed)

    normalizer_fn = None if not config.act_norm else functools.partial(
        spt.layers.act_norm,
        axis=-1 if config.channels_last else -3,
        initializing=is_initializing,
        value_ndims=3,
    )

    # sample z ~ p(z)
    z = net.add('z', spt.Normal(mean=tf.zeros([1, config.z_dim]),
                                logstd=tf.zeros([1, config.z_dim])),
                group_ndims=1, n_samples=n_z)

    # compute the hidden features
    with arg_scope([spt.layers.deconv2d],
                   kernel_size=config.kernel_size,
                   # shortcut_kernel_size=config.shortcut_kernel_size,
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
        h_z = spt.layers.deconv2d(h_z, 64)  # output: (64, 7, 7)
        h_z = spt.layers.deconv2d(h_z, 32, strides=2)  # output: (32, 14, 14)
        h_z = spt.layers.deconv2d(h_z, 32)  # output: (32, 14, 14)
        h_z = spt.layers.deconv2d(h_z, 16, strides=2)  # output: (16, 28, 28)

    # sample x ~ p(x|z)
    x_logits = spt.layers.conv2d(
        h_z, 1, (1, 1), padding='same', name='feature_map_to_pixel',
        channels_last=config.channels_last
    )  # output: (1, 28, 28)
    if n_x is not None:
        x_logits = tf.expand_dims(x_logits, 0)
        multiples = [1 for i in range(len(x_logits.shape))]
        multiples[0] = n_x
        # print(multiples)
        x_logits = tf.tile(x_logits, multiples=multiples)
    x = net.add('x', spt.Bernoulli(logits=x_logits), group_ndims=3)

    return net


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
    results.make_dirs('plotting/sample_mcmc', exist_ok=True)
    results.make_dirs('plotting/train.reconstruct', exist_ok=True)
    results.make_dirs('train_summary', exist_ok=True)

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.int32, shape=(None,) + config.x_shape, name='input_x')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)
    beta = tf.placeholder(dtype=tf.float32, shape=(), name='beta')
    gamma = tf.placeholder(dtype=tf.float32, shape=(), name='gamma')

    # derive the loss for initializing
    with tf.name_scope('initialization'), \
         arg_scope([p_net, q_net], is_initializing=True), \
         spt.utils.scoped_set_config(spt.settings, auto_histogram=False):
        init_q_net = q_net(input_x)
        init_chain = init_q_net.chain(p_net, observed={'x': input_x})
        init_loss = tf.reduce_mean(init_chain.vi.training.sgvb())

    # derive the loss and lower-bound for training
    with tf.name_scope('pretraining'), \
         arg_scope([p_net, q_net], is_training=True):
        pretrain_q_net = q_net(input_x)
        pretrain_p_net = p_net(observed={'x': input_x, 'z': pretrain_q_net['z']})
        pretrain_loss = sgvb_loss(pretrain_p_net, pretrain_q_net, beta, {})
        pretrain_loss += tf.losses.get_regularization_loss()

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([p_net, q_net], is_training=True):
        train_q_net = q_net(input_x)
        train_p_net = p_net(observed={'x': input_x, 'z': train_q_net['z']})
        train_another_p_net = p_net(observed={'z': train_q_net['z']}, n_x=config.train_n_x)
        train_another_pq_net = q_net(train_another_p_net['x'], observed={'z': train_q_net['z']})
        log_p_z = train_another_pq_net['z'].log_prob()
        if config.train_n_x is not None:
            log_p_z = -spt.ops.log_mean_exp(-log_p_z, axis=0)
        log_p_z = (1.0 - gamma) * train_p_net['z'].log_prob() + gamma * log_p_z
        # [batch_size]
        vi = VariationalInference(
            log_joint=train_p_net['x'].log_prob() + log_p_z,
            latent_log_probs=[train_q_net['z'].log_prob()]
        )
        train_loss = (
            tf.reduce_mean(vi.training.sgvb()) +
            tf.losses.get_regularization_loss()
        )

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, n_z=config.test_n_z)
        test_p_net = p_net(observed={'x': input_x, 'z': test_q_net['z']})
        test_another_p_net = p_net(observed={'z': test_q_net['z']}, n_x=config.test_n_x)
        test_another_pq_net = q_net(test_another_p_net['x'], observed={'z': test_q_net['z']})
        log_p_z = test_another_pq_net['z'].log_prob()
        if config.test_n_x is not None:
            log_p_z = -spt.ops.log_mean_exp(-log_p_z, axis=0)
        # [test_n_z, batch_size]
        vi = VariationalInference(
            log_joint=test_p_net['x'].log_prob() + log_p_z,
            latent_log_probs=[test_q_net['z'].log_prob()],
            axis=0
        )
        test_nll = -tf.reduce_mean(vi.evaluation.is_loglikelihood())
        test_lb = tf.reduce_mean(vi.lower_bound.elbo())

    # derive the final nll and logits output for testing
    with tf.name_scope('testing'):
        final_test_q_net = q_net(input_x, n_z=config.final_test_n_z)
        final_test_p_net = p_net(observed={'x': input_x, 'z': final_test_q_net['z']})
        final_test_another_p_net = p_net(observed={'z': final_test_q_net['z']}, n_x=config.final_test_n_x)
        final_test_another_pq_net = q_net(final_test_another_p_net['x'], observed={'z': final_test_q_net['z']})
        log_p_z = final_test_another_pq_net['z'].log_prob()
        if config.final_test_n_x is not None:
            log_p_z = -spt.ops.log_mean_exp(-log_p_z, axis=0)
        # [final_test_n_z, batch_size]
        vi = VariationalInference(
            log_joint=final_test_p_net['x'].log_prob() + log_p_z,
            latent_log_probs=[final_test_q_net['z'].log_prob()],
            axis=0
        )
        final_test_nll = -tf.reduce_mean(vi.evaluation.is_loglikelihood())
        final_test_lb = tf.reduce_mean(vi.lower_bound.elbo())

    # derive the optimizer
    with tf.name_scope('optimizing'):
        params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer_pretrain = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradients(train_loss, params)
        grads_pretrain = optimizer_pretrain.compute_gradients(pretrain_loss, params)
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.apply_gradients(grads)
            pretrain_op = optimizer.apply_gradients(grads_pretrain)

        # derive the plotting function
        with tf.name_scope('plotting'):
            x_plots = tf.reshape(
                bernoulli_as_pixel(p_net(n_z=100)['x']), (-1,) + config.x_shape)
            z_plots = tf.reshape(
                p_net(n_z=1000)['z'], (-1, config.z_dim)
            )
            reconstruct_q_net = q_net(input_x)
            reconstruct_z = reconstruct_q_net['z']
            reconstruct_x = p_net(observed={'z': reconstruct_z})['x']
            reconstruct_plots = tf.reshape(
                bernoulli_as_pixel(reconstruct_x),
                (-1,) + config.x_shape
            )
            z_ph = tf.placeholder(dtype=tf.float32, shape=(None, config.z_dim), name='input_z')

        def plot_samples(loop):
            with loop.timeit('plot_time'):
                # plot samples
                max_mcmc_iterator = 20
                x_0 = session.run(x_plots)
                z_0 = None
                images = np.zeros((100 * max_mcmc_iterator,) + config.x_shape, dtype=np.uint8)
                for mcmc_iterator in range(max_mcmc_iterator):
                    images[mcmc_iterator::max_mcmc_iterator, ...] = x_0.astype(np.uint8)
                    x_samples = reconstruct_sampler.sample(x_0 / 255.0)
                    [x_0, z_0] = session.run([reconstruct_plots, reconstruct_z], feed_dict={input_x: x_samples})

                save_images_collection(
                    images=images,
                    filename='plotting/sample_mcmc/{}.png'.format(loop.epoch),
                    grid_size=(10, 10 * max_mcmc_iterator),
                    results=results,
                    channels_last=config.channels_last,
                )
                images = np.zeros((200,) + config.x_shape, dtype=np.uint8)
                images[::2, ...] = x_0.astype(np.int8)
                index_image = np.zeros(shape=(100,) + config.x_shape, dtype=np.int)
                index_mse = np.zeros(shape=100, dtype=np.float) + np.inf
                for [x] in reconstruct_train_flow:
                    for i in range(len(x)):
                        train_set_image = x[i]
                        mse = np.sum(np.reshape(np.square(x_0 - train_set_image), (100, -1)), axis=-1)
                        mask = (mse < index_mse)
                        index_image[mask] = train_set_image
                        index_mse[mask] = mse[mask]
                images[1::2, ...] = index_image
                save_images_collection(
                    images=images,
                    filename='plotting/sample/{}.png'.format(loop.epoch),
                    grid_size=(10, 20),
                    results=results,
                    channels_last=config.channels_last,
                )

                # plot reconstructs
                for [x] in reconstruct_train_flow:
                    x_samples = reconstruct_sampler.sample(x / 255.0)
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
        spt.datasets.load_fashion_mnist(x_shape=config.x_shape)
    train_flow = bernoulli_flow(
        x_train, config.batch_size, shuffle=True, skip_incomplete=True)
    reconstruct_train_flow = spt.DataFlow.arrays(
        [x_train], 50, shuffle=True, skip_incomplete=True)
    test_flow = bernoulli_flow(
        x_test, config.test_batch_size, sample_now=True)
    reconstruct_sampler = spt.preprocessing.BernoulliSampler()

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        # initialize the network
        for [x] in train_flow:
            print('Network initialized, first-batch loss is {:.6g}.\n'.
                  format(session.run(init_loss, feed_dict={input_x: x})))
            break

        # train the network
        with spt.TrainLoop(params,
                           var_groups=['q_net', 'p_net'],
                           max_epoch=config.max_epoch,
                           max_step=config.max_step,
                           summary_dir=(results.system_path('train_summary')
                                        if config.write_summary else None),
                           summary_graph=tf.get_default_graph(),
                           early_stopping=False,
                           checkpoint_dir=results.system_path('checkpoint'),
                           checkpoint_epoch_freq=100,) as loop:

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
            for epoch in limited(epoch_iterator, config.converge_epoch_begin - loop.epoch):
                step_iterator = MyIterator(loop.iter_steps(train_flow))
                for step, [x] in step_iterator:
                    _, batch_loss = session.run(
                        [pretrain_op, pretrain_loss], feed_dict={
                            input_x: x,
                            beta: min(1., 1.0 * epoch / config.warm_up_epoch)
                        })
                    loop.collect_metrics(train_loss=batch_loss)
                if epoch % config.lr_anneal_epoch_freq == 0:
                    learning_rate.anneal()
                if epoch % config.plot_epoch_freq == 0:
                    plot_samples(loop)
                if epoch % config.test_epoch_freq == 0:
                    with loop.timeit('eval_time'):
                        evaluator.run()
                loop.print_logs()

            # training
            for epoch in epoch_iterator:
                step_iterator = MyIterator(loop.iter_steps(train_flow))
                for step, [x] in step_iterator:
                    gamma_value = min(1.0, 1.0 * (epoch - config.converge_epoch_begin) / config.converge_epoch_length)
                    _, batch_loss = session.run(
                        [train_op, train_loss], feed_dict={
                            input_x: x,
                            gamma: gamma_value
                        })
                    loop.collect_metrics(train_loss=batch_loss)
                if epoch % config.lr_anneal_epoch_freq == 0:
                    learning_rate.anneal()
                if epoch % config.plot_epoch_freq == 0:
                    plot_samples(loop)
                if epoch % config.test_epoch_freq == 0:
                    with loop.timeit('eval_time'):
                        evaluator.run()
                loop.print_logs()

                if epoch == config.max_epoch:
                    evaluator = spt.Evaluator(
                        loop,
                        metrics={'test_nll': final_test_nll, 'test_lb': final_test_lb},
                        inputs=[input_x],
                        data_flow=test_flow,
                        time_metric_name='test_time'
                    )
                    evaluator.events.on(
                        spt.EventKeys.AFTER_EXECUTION,
                        lambda e: results.update_metrics(evaluator.last_metrics_dict)
                    )
                    evaluator.run()
                    loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
