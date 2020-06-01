import numpy as np
import tensorflow as tf

from mwcnn import IWT


def test_iwt():
    x = tf.random.normal([1, 191, 241, 4])
    x_res = IWT()(x)
    x1 = (x[..., 0:1] / 2).numpy()
    x2 = (x[..., 1:2] / 2).numpy()
    x3 = (x[..., 2:3] / 2).numpy()
    x4 = (x[..., 3:4] / 2).numpy()
    x_expected = np.zeros([1, 2*191, 2*241, 1])
    x_expected[:, 0::2, 0::2] = x1 - x2 - x3 + x4
    x_expected[:, 1::2, 0::2] = x1 - x2 + x3 - x4
    x_expected[:, 0::2, 1::2] = x1 + x2 - x3 - x4
    x_expected[:, 1::2, 1::2] = x1 + x2 + x3 + x4
    tf_tester = tf.test.TestCase()
    tf_tester.assertAllClose(x_res, x_expected)
