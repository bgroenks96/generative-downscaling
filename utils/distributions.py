import tensorflow as tf
import tensorflow_probability as tfp

class BernoulliGamma(tfp.distributions.Distribution):
    def __init__(self, logits, alpha, beta, epsilon=1.0E-3, dtype=tf.float32, validate_args=False, allow_nan_stats=False,
                 batch_shape=(None,), name='BernoulliGamma'):
        parameters = dict(locals())
        super(BernoulliGamma, self).__init__(dtype=dtype,
                                             reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
                                             validate_args=validate_args,
                                             allow_nan_stats=allow_nan_stats,
                                             parameters=parameters,
                                             name=name)
        self.epsilon = epsilon
        self.bernoulli = tfp.distributions.Bernoulli(logits=logits, allow_nan_stats=False, dtype=tf.float32)
        self.gamma = tfp.distributions.Gamma(alpha, beta, validate_args=False, allow_nan_stats=False)

    def _log_prob(self, y, name='log_prob', **kwargs):
        y_mask = y > self.epsilon
        return tf.where(y_mask,
                        self.bernoulli.log_prob(y_mask) + self.gamma.log_prob(self.epsilon + tf.math.maximum(0.0, y)),
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
    
    def _batch_shape_tensor(self):
        return self.bernoulli.batch_shape_tensor()
        
    def _event_shape_tensor(self):
        return self.bernoulli.event_shape_tensor()
    
class BernoulliLogNormal(tfp.distributions.Distribution):
    def __init__(self, probs, mu, sigma, dtype=tf.float32, validate_args=False, allow_nan_stats=False,
                 epsilon=1.0E-5, batch_shape=(None,), name='BernoulliGamma'):
        parameters = dict(locals())
        super().__init__(dtype=dtype,
                         reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
                         validate_args=validate_args,
                         allow_nan_stats=allow_nan_stats,
                         parameters=parameters,
                         name=name)
        self.bernoulli = tfp.distributions.Bernoulli(probs=probs, allow_nan_stats=False, dtype=tf.float32)
        self.log_normal = tfp.distributions.LogNormal(loc=mu, scale=sigma, validate_args=False, allow_nan_stats=False)
        self.epsilon = epsilon

    def _log_prob(self, y, name='log_prob', **kwargs):
        y_mask = y > self.epsilon
        return tf.where(y_mask,
                        self.bernoulli.log_prob(y_mask) + self.log_normal.log_prob(self.epsilon + tf.math.maximum(0.0, y)),
                        self.bernoulli.log_prob(0.))
    
    def _sample_n(self, n, seed=None):
        return self.bernoulli.sample(n, seed=seed)*self.log_normal.sample(n, seed=seed)
    
    def _mean(self):
        p = self.bernoulli.probs_parameter()
        mu = self.log_normal.loc
        sigma = self.log_normal.scale
        return p*tf.math.exp(mu + sigma**2 / 2.0)
    
    def _variance(self):
        raise NotImplementedError()
        
    def _batch_shape_tensor(self):
        return self.bernoulli.batch_shape_tensor()
        
    def _event_shape_tensor(self):
        return self.bernoulli.event_shape_tensor()
    
    def _event_shape(self):
        return self.bernoulli.event_shape

class BernoulliExponential(tfp.distributions.Distribution):
    def __init__(self, probs, lam, dtype=tf.float32, validate_args=False, allow_nan_stats=False,
                 epsilon=1.0E-5, batch_shape=(None,), name='BernoulliGamma'):
        parameters = dict(locals())
        super().__init__(dtype=dtype,
                         reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
                         validate_args=validate_args,
                         allow_nan_stats=allow_nan_stats,
                         parameters=parameters,
                         name=name)
        self.bernoulli = tfp.distributions.Bernoulli(probs=probs, allow_nan_stats=False, dtype=tf.float32)
        self.exponential = tfp.distributions.Exponential(lam, validate_args=False, allow_nan_stats=False)
        self.epsilon = epsilon

    def _log_prob(self, y, name='log_prob', **kwargs):
        y_mask = y > self.epsilon
        return tf.where(y_mask,
                        self.bernoulli.log_prob(y_mask) + self.exponential.log_prob(self.epsilon + tf.math.maximum(0.0, y)),
                        self.bernoulli.log_prob(0.))
    
    def _sample_n(self, n, seed=None):
        ber_sample = self.bernoulli.sample(n, seed=seed)
        return tf.where(ber_sample > 0, self.exponential.sample(n, seed=seed),
                        tf.random.uniform((n,), minval=self.epsilon/2, maxval=self.epsilon))
    
    def _mean(self):
        p = self.bernoulli.probs_parameter()
        lam = self.exponential.rate
        return p/lam
    
    def _variance(self):
        raise NotImplementedError()
        
    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

def bernoulli_gamma(bijector=None, axis=-1, epsilon=1.0E-6):
    """
    Returns a joint Bernoulli-Gamma distribution constructor, paramterized by 'params'.
    Parameters p, alpha, beta are selected from the given axis, in that order.
    """
    def _bernoulli_gamma(params: tf.Tensor):
        logits = tf.gather(params, [0], axis=axis)
        alpha = tf.math.log1p(epsilon + tf.math.exp(tf.gather(params, [1], axis=axis)))
        beta = tf.math.log1p(epsilon + tf.math.exp(tf.gather(params, [2], axis=axis)))
        dist = BernoulliGamma(logits, alpha, beta)
        if bijector is not None:
            dist = tfp.distributions.TransformedDistribution(dist, bijector=bijector)
        return dist
    return _bernoulli_gamma

def normal(bijector=None, axis=-1, epsilon=1.0E-6):
    def _normal(params: tf.Tensor):
        mus = tf.gather(params, [0], axis=axis)
        log_sigmas = tf.gather(params, [1], axis=axis)
        dist = tfp.distributions.Normal(loc=mus, scale=epsilon + tf.math.exp(log_sigmas), allow_nan_stats=False)
        if bijector is not None:
            dist = tfp.distributions.TransformedDistribution(dist, bijector=bijector)
        return dist
    return _normal

def logistic(bijector=None, axis=-1, epsilon=1.0E-6):
    def _logistic(params: tf.Tensor):
        mus = tf.gather(params, [0], axis=axis)
        log_scales = tf.gather(params, [1], axis=axis)
        dist = tfp.distributions.Logistic(loc=mus, scale=epsilon + tf.math.exp(log_scales), allow_nan_stats=False)
        if bijector is not None:
            dist = tfp.distributions.TransformedDistribution(dist, bijector=bijector)
        return dist
    return _logistic
    

