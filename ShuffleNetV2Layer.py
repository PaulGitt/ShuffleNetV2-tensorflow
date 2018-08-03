import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python import layers


def shufflenetv2block(inputs):
    inputs1, inputs2 = channel_split(inputs)

    in_filters = int(inputs2.get_shape().as_list()[-1])
    inputs2_conv1 = _conv2d(inputs2, [1, 1, in_filters, in_filters], [1, 1, 1, 1])
    inputs2_bnrelu = BNRelu(inputs2_conv1)
    inputs2_conv2 = _depthwise_conv(inputs2_bnrelu, [1, 1, in_filters, 1],
                                    [1, 1, 1, 1])
    inputs2_bn = BNRelu(inputs2_conv2, RELU=False)
    inputs2_conv3 = _conv2d(inputs2_bn, [1, 1, in_filters, in_filters], [1, 1, 1, 1])
    inputs2_bnrelu2 = BNRelu(inputs2_conv3)

    inputs_concat = tf.concat([inputs1, inputs2_bnrelu2], 3)
    inputs_channelshuffle = channel_shuffle(inputs_concat)
    return inputs_channelshuffle


def shufflenetv2block_withsubsample(inputs):
    in_filters = int(inputs.get_shape().as_list()[-1])
    inputs_part1_conv1 = _depthwise_conv(inputs, [3, 3, in_filters, 1],
                                         [1, 2, 2, 1], padding='VALID')
    inputs_part1_bnrelu = BNRelu(inputs_part1_conv1)
    inputs_part1_conv2 = _conv2d(inputs_part1_bnrelu, [1, 1, in_filters, in_filters],
                                 [1, 1, 1, 1])
    inputs_part1_bn = BNRelu(inputs_part1_conv2, RELU=False)

    inputs_part2_conv1 = _conv2d(inputs, [1, 1, in_filters, in_filters], [1, 1, 1, 1])
    inputs_part2_bnrelu = BNRelu(inputs_part2_conv1)
    inputs_part2_conv2 = _depthwise_conv(inputs_part2_bnrelu, [3, 3, in_filters, 1],
                                         [1, 2, 2, 1], padding='VALID')
    inputs_part2_bn = BNRelu(inputs_part2_conv2, RELU=False)
    inputs_part2_conv3 = _conv2d(inputs_part2_bn, [1, 1, in_filters, in_filters], [1, 1, 1, 1])
    inputs_part2_bnrelu3 = BNRelu(inputs_part2_conv3)

    inputs_concat = tf.concat([inputs_part1_bn, inputs_part2_bnrelu3], 3)
    inputs_channelshuffle = channel_shuffle(inputs_concat)
    return inputs_channelshuffle


def channel_shuffle(inputs, num_groups=8):
    n, h, w, c = inputs.get_shape()
    x = tf.reshape(inputs, [-1, h, w, num_groups, c // num_groups])
    x_transponse = tf.transpose(x, [0, 1, 2, 4, 3])
    x_reshape = tf.reshape(x_transponse, [-1, h, w, c])
    return x_reshape


def channel_split(inputs, num_splits=2):
    c = inputs.get_shape()[3]
    input1, input2 = tf.split(inputs, [int(c // num_splits), int(c // num_splits)], axis=3)
    return input1, input2


def _depthwise_conv(inputs, weights_shape, strides, padding='SAME'):
    weights = tf.Variable(tf.random_normal(shape=weights_shape, dtype=tf.float32))
    return nn.depthwise_conv2d(inputs, weights, strides,
                               padding=padding)


def _conv2d(inputs, weights_shape, strides, padding='SAME'):
    weights = tf.Variable(tf.random_normal(shape=weights_shape, dtype=tf.float32))
    return nn.conv2d(inputs, weights, strides, padding=padding)


def BNRelu(inputs, RELU=True):
    batchnorm = layers.batch_normalization(inputs)
    if RELU:
        relu = nn.relu(batchnorm)
        return relu
    else:
        return batchnorm
