import tensorflow as tf
import numpy as np
import maddpg.maddpgAlgor.common.tf_util as U
from tensorflow.python.ops import math_ops
from maddpg.multiagent.multi_discrete import MultiDiscrete
from tensorflow.python.ops import nn

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def logp(self, x):
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError

class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)

class SoftCategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat #5
    def pdclass(self):
        return SoftCategoricalPd
    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return [self.ncat]
    def sample_dtype(self):
        return tf.float32


class SoftCategoricalPd(Pd):
    def __init__(self, logits): # logits = p_func output
        self.logits = logits
    def flatparam(self):
        return self.logits

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return U.softmax(self.logits - tf.log(-tf.log(u)), axis=-1)


def make_pdtype(ac_space):
    return SoftCategoricalPdType(ac_space.n) ###### trainSheepChasing SoftCategoricalPdType(5)
