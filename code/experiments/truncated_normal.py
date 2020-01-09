import contextlib

import tensorflow as tf
import zhusuan.distributions as zd

from tfsnippet.distributions.utils import compute_density_immediately, reduce_group_ndims
from zhusuan.utils import add_name_scope
from tfsnippet.utils import settings, get_default_scope_name

from tfsnippet.distributions.wrapper import ZhuSuanDistribution


class TruncatedNormal(ZhuSuanDistribution):
    """
    Univariate Normal distribution.

    See Also:
        :class:`tfsnippet.distributions.Distribution`,
        :class:`zhusuan.distributions.Distribution`,
        :class:`zhusuan.distributions.Normal`
    """

    def __init__(self, mean, std=None, logstd=None, is_reparameterized=True,
                 check_numerics=None):
        """
        Construct the :class:`Normal`.

        Args:
            mean: A `float` tensor, the mean of the Normal distribution.
                Should be broadcastable against `std` / `logstd`.
            std: A `float` tensor, the standard deviation of the Normal
                distribution.  Should be positive, and broadcastable against
                `mean`.  One and only one of `std` or `logstd` should be
                specified.
            logstd: A `float` tensor, the log standard deviation of the Normal
                distribution.  Should be broadcastable against `mean`.
            is_reparameterized (bool): Whether or not the gradients can
                be propagated through parameters? (default :obj:`True`)
            check_numerics (bool): Whether or not to check numerical issues.
                Default to ``tfsnippet.settings.check_numerics``.
        """
        if check_numerics is None:
            check_numerics = settings.check_numerics
        super(TruncatedNormal, self).__init__(zd.Normal(
            mean=mean,
            std=std,
            logstd=logstd,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
        ))

    @property
    def mean(self):
        """Get the mean of the Normal distribution."""
        return self._distribution.mean

    @property
    def logstd(self):
        """Get the log standard deviation of the Normal distribution."""
        return self._distribution.logstd

    @property
    def std(self):
        """Get the standard deviation of the Normal distribution."""
        return self._distribution.std

    @add_name_scope
    def _sample(self, n_samples=None):
        """
        sample(n_samples=None)

        Return samples from the distribution. When `n_samples` is None (by
        default), one sample of shape ``batch_shape + value_shape`` is
        generated. For a scalar `n_samples`, the returned Tensor has a new
        sample dimension with size `n_samples` inserted at ``axis=0``, i.e.,
        the shape of samples is ``[n_samples] + batch_shape + value_shape``.

        :param n_samples: A 0-D `int32` Tensor or None. How many independent
            samples to draw from the distribution.
        :return: A Tensor of samples.
        """
        if n_samples is None:
            samples = self.__sample(n_samples=1)
            return tf.squeeze(samples, axis=0)
        elif isinstance(n_samples, int):
            return self.__sample(n_samples)
        else:
            n_samples = tf.convert_to_tensor(n_samples, dtype=tf.int32)
            _assert_rank_op = tf.assert_rank(
                n_samples, 0,
                message="n_samples should be a scalar (0-D Tensor).")
            with tf.control_dependencies([_assert_rank_op]):
                samples = self.__sample(n_samples)
            return samples

    def __sample(self, n_samples):
        mean, std = self.mean, self.std
        if not self.is_reparameterized:
            mean = tf.stop_gradient(mean)
            std = tf.stop_gradient(std)
        shape = tf.concat([[n_samples], self.batch_shape], 0)
        samples = tf.truncated_normal(shape, dtype=self.dtype) * std + mean
        static_n_samples = n_samples if isinstance(n_samples, int) else None
        samples.set_shape(
            tf.TensorShape([static_n_samples]).concatenate(
                self.get_batch_shape()))
        return samples

    def sample(self, n_samples=None, is_reparameterized=None, group_ndims=0,
               compute_density=None, name=None):
        from tfsnippet.stochastic import StochasticTensor

        self._validate_sample_is_reparameterized_arg(is_reparameterized)

        if is_reparameterized is False and self.is_reparameterized:
            @contextlib.contextmanager
            def set_is_reparameterized():
                try:
                    self._distribution._is_reparameterized = False
                    yield False
                finally:
                    self._distribution._is_reparameterized = True
        else:
            @contextlib.contextmanager
            def set_is_reparameterized():
                yield self.is_reparameterized

        with tf.name_scope(name=name, default_name='sample'):
            with set_is_reparameterized() as is_reparameterized:
                samples = self._sample(n_samples=n_samples)
                t = StochasticTensor(
                    distribution=self,
                    tensor=samples,
                    n_samples=n_samples,
                    group_ndims=group_ndims,
                    is_reparameterized=is_reparameterized,
                )
                if compute_density:
                    compute_density_immediately(t)
                return t

    def log_prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name=name,
                           default_name=get_default_scope_name('log_prob', self)):
            given = self._distribution._check_input_shape(given)
            truncated_area = 0.95449974
            log_prob = self._distribution._log_prob(given)
            log_prob = log_prob - tf.log(truncated_area)
            return reduce_group_ndims(tf.reduce_sum, log_prob, group_ndims)
