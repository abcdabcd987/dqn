import tensorflow as tf
import numpy
import abc


def _variable(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def _weight_initializer():
    # return tf.random_uniform_initializer(minval=-0.05, maxval=0.05, dtype=tf.float32)
    return tf.truncated_normal_initializer(mean=0.0, stddev=0.01)


def _bias_initializer():
    return tf.constant_initializer(0.0)


class Layer(object):
    __metaclass__ = abc.ABCMeta

    @property
    def shape(self):
        return self._shape

    @property
    def name(self):
        return self._name

    @property
    def params(self):
        return self._params

    @abc.abstractmethod
    def __str__(str):
        pass

    @abc.abstractmethod
    def forward(self, prev, x):
        '''this method should set self._shape'''
        pass

    def __init__(self, name):
        self._name = name
        self._shape = None
        self._params = []


class InputLayer(Layer):
    def __init__(self):
        Layer.__init__(self, name='input')

    def __str__(self):
        return 'InputLayer() shape: {self._shape:r}'.format(self=self)

    def forward(self, _, x):
        self._shape = x.get_shape().as_list()
        return x


class FCLayer(Layer):
    def __init__(self, name, dim, act):
        Layer.__init__(self, name)
        self._dim = dim
        self._act = act

    def __str__(self):
        return 'FCLayer(name={self._name}, dim={self._dim:d}, act={self._act:r}) '\
               'shape: {self._shape:r}'.format(self=self)

    def forward(self, prev, x):
        with tf.variable_scope(self._name):
            w = _variable('weight', [prev.shape[1], self._dim], _weight_initializer())
            b = _variable('bias', [self._dim], _bias_initializer())
            out = tf.matmul(x, w) + b
            if self._act is not None:
                out = self._act(out, name='act')
        self.l2_loss = tf.nn.l2_loss(w)
        self.l1_loss = tf.reduce_sum(tf.abs(w))
        self._shape = out.get_shape().as_list()
        self._params = [w, b]
        return out


class Conv2DLayer(Layer):
    def __init__(self, name, ksize, kernels, strides, padding, act):
        Layer.__init__(self, name)
        self._ksize = ksize
        self._kernels = kernels
        self._strides = strides
        self._padding = padding
        self._act = act

    def __str__(self):
        return 'Conv2DLayer(name={self._name}, ksize={self._ksize:r}, '\
               'kernels={self._kernels:d}, strides={self._strides:r}, '\
               'padding={self._padding:r}) shape: {self._shape:r}'.format(self=self)

    def forward(self, prev, x):
        filter_shape = (self._ksize[0], self._ksize[1], prev.shape[3], self._kernels)
        strides = (1, self._strides[0], self._strides[1], 1)
        with tf.variable_scope(self._name):
            w = _variable('weight', filter_shape, _weight_initializer())
            b = _variable('bias', [self._kernels], _bias_initializer())
            out = tf.nn.conv2d(x, w, strides=strides, padding=self._padding)
            out = tf.nn.bias_add(out, b)
            if self._act:
                out = self._act(out, name='act')
        self._shape = out.get_shape().as_list()
        self._params = [w, b]
        return out


class MaxPoolLayer(object):
    def __init__(self, ksize, padding='VALID'):
        Layer.__init__(self, name='pool')
        self._ksize = ksize
        self._padding = padding
        self._shape = None

    def __str__(self):
        return 'MaxPoolLayer(ksize={self._ksize:r}, padding={self._padding:r}) shape: {self._shape:r}'.format(self=self)

    def forward(self, prev, x):
        ksize = (1, self._ksize[0], self._ksize[1], 1)
        out = tf.nn.max_pool(x, ksize, ksize, self._padding)
        self._shape = out.get_shape().as_list()
        return out


class DropoutLayer(object):
    ph_prob_keep = None

    def __init__(self):
        Layer.__init__(self, name='dropout')
        if Dropout.ph_prob_keep is None:
            Dropout.ph_prob_keep = tf.placeholder(tf.float32)

    def __str__(self):
        return 'DropoutLayer() shape: {self._shape:r}'.format(self=self)

    def forward(self, prev, x):
        out = tf.nn.dropout(x, Dropout.ph_prob_keep)
        self._shape = out.get_shape().as_list()
        return out


class FlattenLayer(Layer):
    def __init__(self):
        Layer.__init__(self, name='flatten')

    def __str__(self):
        return 'FlattenLayer() shape: {self._shape:r}'.format(self=self)

    def forward(self, prev, x):
        shape = x.get_shape().as_list()
        dim = numpy.prod(shape[1:])
        out = tf.reshape(x, [-1, dim])
        self._shape = out.get_shape().as_list()
        return out


def forward_all_layers(layers, x):
    l1_loss, l2_loss = 0., 0.
    prev, out = None, x
    for layer in layers:
        if hasattr(layer, 'l2_loss'):
            l2_loss += layer.l2_loss
        if hasattr(layer, 'l1_loss'):
            l1_loss += layer.l1_loss
        out = layer.forward(prev, out)
        prev = layer
    return out, l1_loss, l2_loss


def get_all_parameters(layers):
    params = []
    for layer in layers:
        params.extend(layer.params)
    return params
