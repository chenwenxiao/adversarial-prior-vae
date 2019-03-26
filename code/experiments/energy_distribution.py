import tensorflow as tf
import tfsnippet as spt
from tfsnippet.distributions.utils import compute_density_immediately

__all__ = ['EnergyDistribution']


class EnergyDistribution(spt.Distribution):
    """
    A distribution derived from an energy function `U(x)` and a generator
    function `x = G(z)`, where `p(x) = exp(-U(x)) / Z`.
    """

    def __init__(self, pz, G, U, log_Z=0., mcmc_on_z=False, mcmc_on_x=False):
        """
        Construct a new :class:`EnergyDistribution`.

        Args:
            pz (spt.Distribution): The base distribution `p(z)`.
            G: The function `x = G(z)`.
            U: The function `U(x)`.
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
        self._U = U
        with tf.name_scope('log_Z', values=[log_Z]):
            self._log_Z = tf.maximum(log_Z, -20)

    @property
    def pz(self):
        return self._pz

    @property
    def G(self):
        return self._G

    @property
    def U(self):
        return self._U

    @property
    def Z(self):
        return self._Z

    @property
    def log_Z(self):
        return self._log_Z

    def log_prob(self, given, group_ndims=0, name=None):
        given = tf.convert_to_tensor(given)
        with tf.name_scope(name,
                           default_name=spt.utils.get_default_scope_name(
                               'log_prob', self),
                           values=[given]):
            energy = self.U(given)
            log_px = -energy - self.log_Z
            log_px = spt.distributions.reduce_group_ndims(
                tf.reduce_sum, log_px, group_ndims=group_ndims)
            log_px.energy = energy

        return log_px

    def sample(self, n_samples=None, group_ndims=0, is_reparameterized=None,
               compute_density=None, name=None):
        self._validate_sample_is_reparameterized_arg(is_reparameterized)
        if is_reparameterized is None:
            is_reparameterized = self.is_reparameterized

        with tf.name_scope(name,
                           default_name=spt.utils.get_default_scope_name(
                               'sample', self)):
            z = self.pz.sample(
                n_samples=n_samples, is_reparameterized=is_reparameterized,
                compute_density=False
            )
            x = self.G(z)

            if self.is_reparameterized and not is_reparameterized:
                x = tf.stop_gradient(x)

            t = spt.StochasticTensor(
                distribution=self,
                tensor=x,
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized
            )
            if compute_density:
                compute_density_immediately(t)
            t.z = z
            t.z_ = self.pz.sample(
                n_samples=n_samples, is_reparameterized=is_reparameterized,
                compute_density=False
            )
        return t
