import numpy as np
import tensorflow as tf
import unittest
from bcsd import QuantileMap

class TestQuantileMap(unittest.TestCase):
    def setUp(self):
        tf.compat.v1.enable_eager_execution()

    def test_simple_mapping_2d(self):
        num_quantiles = 10
        qmap = QuantileMap(num_quantiles)
        x = tf.constant([[i] for i in range(num_quantiles)], dtype=tf.float32)
        assert x.shape == [num_quantiles, 1], x.shape
        y = tf.constant([[i+1] for i in range(num_quantiles)], dtype=tf.float32)
        assert y.shape == [num_quantiles, 1], y.shape
        qmap.fit(x, y)
        y_pred = qmap.predict(x)
        assert np.array_equal(y, y_pred), f'expected {y} but got {y_pred}'

    def test_gaussian_mapping_3d(self):
        num_quantiles = 10000
        qmap = QuantileMap(num_quantiles)
        x = tf.random.normal((1000, 32, 32))
        y = tf.random.normal((1000, 32, 32), mean=2.0, stddev=0.5)
        qmap.fit(x, y)
        y_pred = qmap.predict(x, batch_size=10)
        assert np.isclose(2.0, np.mean(y_pred), atol=0.001)
        assert np.isclose(0.5, np.std(y_pred), atol=0.001)
