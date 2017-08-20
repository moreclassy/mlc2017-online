# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import tensorflow as tf
import utils

import tensorflow.contrib.slim as slim

"""Contains the base class for models."""
class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    raise NotImplementedError()

class Convolution(BaseModel):
    
  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
    net = slim.conv2d(model_input, 20, [5,5], scope='conv1')
    net = slim.max_pool2d(net, [2,2], scope='pool1')
    net = slim.conv2d(net, 50, [5,5], scope='conv2')
    net = slim.max_pool2d(net, [2,2], scope='pool2')
    net = slim.flatten(net, scope='flatten3')
    net = slim.fully_connected(net, 500, scope='fc4')
    output = slim.fully_connected(
        net, num_classes, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class Convolution2(BaseModel):
    
    def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
        net = slim.repeat(model_input, 2, slim.conv2d, 64, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.fully_connected(net, 4096)
        net = slim.dropout(net, 0.5)
        net = slim.fully_connected(net, 4096)
        net = slim.dropout(net, 0.5)
        net = slim.flatten(net)
        output = slim.fully_connected(
          net, num_classes, activation_fn=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
        return {"predictions": output}

class Convolution3(BaseModel):
    
    def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
        net = slim.repeat(model_input, 2, slim.conv2d, 64, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.fully_connected(net, 4096)
        net = slim.dropout(net, 0.5)
        net = slim.flatten(net)
        output = slim.fully_connected(
          net, num_classes, activation_fn=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
        return {"predictions": output}

class Convolution4(BaseModel):
    
    def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
        net = slim.flatten(model_input)
        net = tf.layers.conv2d(
            inputs=model_input,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net = slim.fully_connected(net, 100)
        net = slim.dropout(net, 0.5)
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        net = tf.layers.conv2d(
            inputs=net,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        net = slim.fully_connected(net, 100)
        net = slim.dropout(net, 0.5)
        net = slim.flatten(net)
        output = slim.fully_connected(
            net, num_classes, activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
        return {"predictions": output}

class Convolution5(BaseModel):
    
    def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
        net = slim.flatten(net)
        net = slim.repeat(model_input, 2, slim.conv2d, 64, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3])
        net = slim.max_pool2d(net, [2, 2])
        net = slim.fully_connected(net, 4096)
        net = slim.dropout(net, 0.5)
        net = slim.fully_connected(net, 4096)
        net = slim.dropout(net, 0.5)
        net = slim.flatten(net)
        output = slim.fully_connected(
          net, num_classes, activation_fn=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
        return {"predictions": output}

class LogisticModel(BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      num_classes: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    net = slim.flatten(model_input)
    output = slim.fully_connected(
        net, num_classes, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}
