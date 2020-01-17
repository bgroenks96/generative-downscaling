import tensorflow as tf
import tensorflow_probability as tfp

class BernoulliGamma(tfp.distributions.Distribution):
    def __init__(self, logits, alpha, beta, dtype=tf.float32, validate_args=False, allow_nan_stats=False,
                 batch_shape=(None,), name='BernoulliGamma'):
        parameters = dict(locals())
        super(BernoulliGamma, self).__init__(dtype=dtype,
                                             reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
                                             validate_args=validate_args,
                                             allow_nan_stats=allow_nan_stats,
                                             parameters=parameters,
                                             name=name)
        self.bernoulli = tfp.distributions.Bernoulli(logits=logits, allow_nan_stats=False, dtype=tf.float32)
        self.gamma = tfp.distributions.Gamma(alpha, beta, validate_args=False, allow_nan_stats=False)

    def _log_prob(self, y, epsilon=1.0E-6, name='log_prob', **kwargs):
        y_mask = y > 0
        return tf.where(y_mask,
                        self.bernoulli.log_prob(y_mask) + self.gamma.log_prob(epsilon + tf.math.maximum(0.0, y)),
                        self.bernoulli.log_prob(y_mask))
    
    def _sample_n(self, n, seed=None):
        return self.bernoulli.sample(n, seed=seed)*self.gamma.sample(n, seed=seed)
    
    def _mean(self):
        # mean: pα/β
        p = self.bernoulli.probs_parameter()
        alpha = self.gamma.concentration
        beta = self.gamma.rate
        return p*alpha / beta
    
    def _variance(self):
        # variance: pα[1 + (1 – p)α]/β^2
        p = self.bernoulli.probs_parameter()
        alpha = self.gamma.concentration
        beta = self.gamma.rate
        return p*alpha*(1.0 + (1.0 - p)*alpha) / beta**2

def bernoulli_gamma(params: tf.Tensor, bijector=tfp.bijectors.Identity(), axis=-1, epsilon=1.0E-6):
    """
    Returns a join Bernoulli-Gamma distribution, paramterized by 'params'.
    Parameters p, alpha, beta are selected from the given axis, in that order.
    """
    logits = tf.gather(params, 0, axis=axis)
    alpha = tf.math.log1p(epsilon + tf.math.exp(tf.gather(params, 1, axis=axis)))
    beta = tf.math.log1p(epsilon + tf.math.exp(tf.gather(params, 2, axis=axis)))
    base_dist BernoulliGamma(logits, alpha, beta)
    transformed_dist = tfp.distributions.TransformedDistribution(base_dist, bijector=bijector)
    return transformed_dist

def normal(params: tf.Tensor, bijector=tfp.bijectors.Identity(), axis=-1, epsilon=1.0E-6):
    mus = tf.gather(params, 0, axis=axis)
    log_sigmas = tf.gather(params, 1, axis=axis)
    #shape = tf.shape(mus)
    #mus = tf.reshape(mus, (-1, tf.math.reduce_prod(shape[1:])))
    #log_sigmas = tf.reshape(log_sigmas, (-1, tf.math.reduce_prod(shape[1:])))
    base_dist = tfp.distributions.Normal(loc=mus, scale=epsilon + tf.math.exp(log_sigmas), allow_nan_stats=False)
    transformed_dist = tfp.distributions.TransformedDistribution(base_dist, bijector=bijector)
    return transformed_dist
    

