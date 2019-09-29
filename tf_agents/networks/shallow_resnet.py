# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet50 model for Keras.

Adapted from tf.keras.applications.resnet50.ResNet50().
This is ResNet model version 1.5.

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.networks import network

from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers

import gin

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def _gen_l2_regularizer(use_l2_regularizer=True):
    return regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None


@gin.configurable
class ShallowResnet(network.Network):
    """Feed Forward network with CNN and FNN layers."""

    def __init__(self,
                 input_tensor_spec,
                 num_classes,
                 blocks,
                 num_filters_per_block,
                 dtype='float32',
                 batch_size=None,
                 use_l2_regularizer=True,
                 softmax=True,
                 name='ShallowResnet'):

        self._num_classes = num_classes
        self._blocks = blocks
        self._num_filters_per_block = num_filters_per_block
        self._dtype = dtype
        self._batch_size = batch_size
        self._use_l2_regularizer = use_l2_regularizer
        self._softmax = softmax
        self._name = name

        super(ShallowResnet, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

        self._resnet_layers = dict()

        self._build_resnet(num_classes,
              blocks,
              num_filters_per_block,
              dtype='float32',
              batch_size=None,
              use_l2_regularizer=True,
              reduce_mean=False,
              softmax=True)


    def call(self, observation):

        with tf.compat.v1.variable_scope(self._name, reuse=tf.compat.v1.AUTO_REUSE):
            states = self.shallow_resnet(observation,
                                       self._num_classes,
                                       self._blocks,
                                       self._num_filters_per_block,
                                       dtype=self._dtype,
                                       batch_size=None,
                                       use_l2_regularizer=True,
                                       softmax=True)
        return states

    def _build_identity_block(self, kernel_size,
                       filters,
                       stage,
                       block,
                       use_l2_regularizer=True):

        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        layer_name = conv_name_base + '2a'
        self._resnet_layers[layer_name] = layers.Conv2D(
            filters1, (1, 1),
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            name=layer_name)

        layer_name = bn_name_base + '2a'
        self._resnet_layers[layer_name] = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=layer_name)

        layer_name = conv_name_base + '2b'
        self._resnet_layers[layer_name] = layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            name=layer_name)

        layer_name = bn_name_base + '2b'
        self._resnet_layers[layer_name] = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=layer_name)

        layer_name = conv_name_base + '2c'
        self._resnet_layers[layer_name] = layers.Conv2D(
            filters3, (1, 1),
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            name=layer_name)

        layer_name = bn_name_base + '2c'
        self._resnet_layers[layer_name] = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=layer_name)

        return

    def _build_conv_block(self, kernel_size,
                   filters,
                   stage,
                   block,
                   strides=(2, 2),
                   use_l2_regularizer=True):

        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        layer_name = conv_name_base + '2a'
        self._resnet_layers[layer_name] = layers.Conv2D(
            filters1, (1, 1),
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            name=layer_name)

        layer_name = bn_name_base + '2a'
        self._resnet_layers[layer_name] = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=layer_name)

        layer_name = conv_name_base + '2b'
        self._resnet_layers[layer_name] = layers.Conv2D(
            filters2,
            kernel_size,
            strides=strides,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            name=layer_name)

        layer_name = bn_name_base + '2b'
        self._resnet_layers[layer_name] = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=layer_name)

        layer_name = conv_name_base + '2c'
        self._resnet_layers[layer_name] = layers.Conv2D(
            filters3, (1, 1),
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            name=layer_name)

        layer_name = bn_name_base + '2c'
        self._resnet_layers[layer_name] = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=layer_name)

        layer_name = conv_name_base + '1'
        self._resnet_layers[layer_name] = layers.Conv2D(
            filters3, (1, 1),
            strides=strides,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            name=layer_name)

        layer_name = bn_name_base + '1'
        self._resnet_layers[layer_name] = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=layer_name)

        return

    def identity_block(self, input_tensor,
                       stage,
                       block):
        """The identity block is the block that has no conv layer at shortcut.

        Args:
          input_tensor: input tensor
          kernel_size: default 3, the kernel size of middle conv layer at main path
          filters: list of integers, the filters of 3 conv layer at main path
          stage: integer, current stage label, used for generating layer names
          block: 'a','b'..., current block label, used for generating layer names
          use_l2_regularizer: whether to use L2 regularizer on Conv layer.

        Returns:
          Output tensor for the block.
        """
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = self._resnet_layers[conv_name_base + '2a'](input_tensor)

        x = self._resnet_layers[bn_name_base + '2a']( x)
        x = layers.Activation('relu')(x)

        x = self._resnet_layers[conv_name_base + '2b'](x)
        x = self._resnet_layers[bn_name_base + '2b'](x)
        x = layers.Activation('relu')(x)

        x = self._resnet_layers[conv_name_base + '2c'](x)
        x = self._resnet_layers[bn_name_base + '2c'](x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)

        return x

    def conv_block(self, input_tensor,
                   stage,
                   block):
        """A block that has a conv layer at shortcut.

        Note that from stage 3,
        the second conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well

        Args:
          input_tensor: input tensor
          kernel_size: default 3, the kernel size of middle conv layer at main path
          filters: list of integers, the filters of 3 conv layer at main path
          stage: integer, current stage label, used for generating layer names
          block: 'a','b'..., current block label, used for generating layer names
          strides: Strides for the second conv layer in the block.
          use_l2_regularizer: whether to use L2 regularizer on Conv layer.

        Returns:
          Output tensor for the block.
        """
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = self._resnet_layers[conv_name_base + '2a'](input_tensor)
        x = self._resnet_layers[bn_name_base + '2a'](x)
        x = layers.Activation('relu')(x)

        x = self._resnet_layers[conv_name_base + '2b'](x)
        x = self._resnet_layers[bn_name_base + '2b'](x)
        x = layers.Activation('relu')(x)

        x = self._resnet_layers[conv_name_base + '2c'](x)
        x = self._resnet_layers[bn_name_base + '2c'](x)

        shortcut = self._resnet_layers[conv_name_base + '1'](input_tensor)
        shortcut = self._resnet_layers[bn_name_base + '1'](shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def _build_resnet(self,
              num_classes,
              blocks,
              num_filters_per_block,
              dtype='float32',
              batch_size=None,
              use_l2_regularizer=True,
              reduce_mean=False,
              softmax=True):

        if backend.image_data_format() == 'channels_first':
            bn_axis = 1
        else:  # channels_last
            bn_axis = 3

        # x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
        self._resnet_layers['conv1'] = layers.Conv2D(
            64, (3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            name='conv1')

        self._resnet_layers['bn_conv1'] = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name='bn_conv1')

        for i, (num_block, num_filter) in enumerate(zip(blocks, num_filters_per_block)):
            stage_id = i + 1

            self._build_conv_block(
                3, [num_filter, num_filter, num_filter],
                stage=stage_id,
                block='a_',
                strides=(1, 1),
                use_l2_regularizer=use_l2_regularizer)

            for j in range(num_block):
                self._build_identity_block(
                    3, [num_filter, num_filter, num_filter],
                    stage=stage_id,
                    block='b_' + str(j),
                    use_l2_regularizer=use_l2_regularizer)
                self._build_identity_block(
                    3, [num_filter, num_filter, num_filter],
                    stage=stage_id,
                    block='c_' + str(j),
                    use_l2_regularizer=use_l2_regularizer)

        return

    def shallow_resnet(self, img_input,
                       num_classes,
                       blocks,
                       num_filters_per_block,
                       dtype='float32',
                       batch_size=None,
                       use_l2_regularizer=True,
                       reduce_mean=False,
                       softmax=True):
        """Instantiates the ResNet50 architecture.

        Args:
          num_classes: `int` number of classes for image classification.
          dtype: dtype to use float32 or float16 are most common.
          batch_size: Size of the batches for each step.
          use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.

        Returns:
            A tensor of final layer
        """
        # input_shape = (224, 224, 3)
        # img_input = layers.Input(
        #     shape=input_shape, dtype=dtype, batch_size=batch_size)

        if backend.image_data_format() == 'channels_first':
            x = layers.Lambda(
                lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
                name='transpose')(
                img_input)
            bn_axis = 1
        else:  # channels_last
            x = img_input
            bn_axis = 3

        # x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
        x = self._resnet_layers['conv1'](x)
        x = self._resnet_layers['bn_conv1'](x)
        x = layers.Activation('relu')(x)
        # x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        for i, (num_block, num_filter) in enumerate(zip(blocks, num_filters_per_block)):
            stage_id = i + 1
            x = self.conv_block(x, stage=stage_id, block='a_')

            for j in range(num_block):
                x = self.identity_block(
                    x, stage=stage_id, block='b_' + str(j))
                x = self.identity_block(
                    x, stage=stage_id, block='c_' + str(j))

        rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]

        if reduce_mean:
            x = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')(x)

        # TODO(reedwm): Remove manual casts once mixed precision can be enabled with a
        # single line of code.
        x = backend.cast(x, 'float32')

        # =============================================================================
        #     if softmax:
        #         x = layers.Activation('softmax')(x)
        # =============================================================================

        # Create model.
        return x
