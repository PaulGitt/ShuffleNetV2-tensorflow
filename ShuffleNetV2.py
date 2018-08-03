# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python import layers
from ShuffleNetV2Layer import _conv2d, shufflenetv2block, \
    shufflenetv2block_withsubsample
from tflearn.layers.estimator import regression
import tflearn
from tflearn.layers.core import input_data
import tflearn.datasets.oxflower17 as oxflower17

X, Y = oxflower17.load_data(one_hot=True)

tf.reset_default_graph()

# Parameters
BatchSize = 32
input_size = 224
x = input_data(shape=[None, input_size, input_size, 3])
Epochs = 10
learning_rate = 1e-4

# Network Structure
x_conv = _conv2d(x, [3, 3, 3, 24], [1, 2, 2, 1], padding='VALID')
# print(x_conv.shape)
x_maxpool = layers.max_pooling2d(x_conv, 3, 2)
# print(x_maxpool.shape)

# Stage2
stage2_block = shufflenetv2block_withsubsample(x_maxpool)
for i in range(3):
    stage2 = stage2_block
    stage2_block = shufflenetv2block(stage2)

# Stage3
stage3_block = shufflenetv2block_withsubsample(stage2_block)
for i in range(7):
    stage3 = stage3_block
    stage3_block = shufflenetv2block(stage3)

# Stage4
stage4_block = shufflenetv2block_withsubsample(stage3_block)
for i in range(3):
    stage4 = stage4_block
    stage4_block = shufflenetv2block(stage4)

# print(stage4_block.shape)
fn_conv = _conv2d(stage4_block, [1, 1, 192, 1024], [1, 1, 1, 1])

kh, kw = fn_conv.get_shape()[1], fn_conv.get_shape()[2]
global_avgpool = tf.reduce_mean(layers.average_pooling2d(fn_conv, [kh, kw], 1), [1, 2])
# flattern = layers.flatten(global_avgpool)
fc = layers.dense(global_avgpool, 17, activation=nn.softmax)

network = regression(fc, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=learning_rate)

# Training
model = tflearn.DNN(network, checkpoint_path='ShuffleNetV2',
                    tensorboard_dir='./log/ShuffleNetV2')
model.fit(X, Y, n_epoch=Epochs, shuffle=True,
          show_metric=True, batch_size=BatchSize,
          run_id='ShuffleNetV2_oxflowers17')
