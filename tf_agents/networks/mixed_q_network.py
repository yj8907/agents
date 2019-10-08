# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from tf_agents.networks import encoding_network
from tf_agents.networks import shallow_resnet
from tf_agents.networks import network

from tensorflow.python.framework import tensor_spec
import numpy as np

def validate_specs(action_spec, observation_spec):
    """Validates the spec contains a single action."""
    # del observation_spec  # not currently validated

    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
        print('Mixed Q Network supports action_specs with more than one action.')

    for spec in flat_action_spec:
        if spec.shape not in [(), (1,)]:
            raise ValueError(
                'Network only supports action_specs with shape in [(), (1,)])')


@gin.configurable
class MixedQNetwork(network.Network):
    """Feed Forward network."""

    def __init__(self,
                 input_tensor_specs,
                 action_specs,
                 spatial_names=("screen","minimap"),
                 structured_names=("structured",),
                 mask_split_fn=None,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=(75, 40),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='MixedQNetwork'):
        """Creates an instance of `QNetwork`.

        Args:
          input_tensor_specs: A dict of list or tuple of nested `tensor_spec.TensorSpec` representing the
            input observations. Observation specs in this case are derived from Starcraft2
            observation wrapper. There should be multiple BoundedTensorSpec objects per key
            for ("screen", "minimap"), representing categorical states on each location.
          action_specs: A dict of dict of nested of `tensor_spec.BoundedTensorSpec` representing the
            actions It includes keys: screen, minimap, structured. There should be only one BoundedTensorSpec
            paer key.
            Action specs in this case are derived from Starcraft2 action wrapper.
          previous_action_spec: A dict of dict of nested of `tensor_spec.BoundedTensorSpec` representing the
            actions It includes keys: screen, minimap, structrued. There should be only one BoundedTensorSpec
            paer key.
            Action specs in this case are derived from Starcraft2 action wrapper.
          mask_split_fn: A function used for masking valid/invalid actions with each
            state of the environment. The function takes in a full observation and
            returns a tuple consisting of 1) the part of the observation intended as
            input to the network and 2) the mask. An example mask_split_fn could be
            as simple as:
            ```
            def mask_split_fn(observation):
              return observation['network_input'], observation['mask']
            ```
            If None, masking is not applied.
          preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
            representing preprocessing for the different observations.
            All of these layers must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
          preprocessing_combiner: (Optional.) A keras layer that takes a flat list
            of tensors and combines them. Good options include
            `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
            This layer must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
          conv_layer_params: Optional list of convolution layers parameters, where
            each item is a length-three tuple indicating (filters, kernel_size,
            stride).
          fc_layer_params: Optional list of fully_connected parameters, where each
            item is the number of units in the layer.
          dropout_layer_params: Optional list of dropout layer parameters, where
            each item is the fraction of input units to drop. The dropout layers are
            interleaved with the fully connected layers; there is a dropout layer
            after each fully connected layer, except if the entry in the list is
            None. This list must have the same length of fc_layer_params, or be
            None.
          activation_fn: Activation function, e.g. tf.keras.activations.relu.
          kernel_initializer: Initializer to use for the kernels of the conv and
            dense layers. If none is provided a default variance_scaling_initializer
          batch_squash: If True the outer_ranks of the observation are squashed into
            the batch dimension. This allow encoding networks to be used with
            observations with shape [BxTx...].
          dtype: The dtype to use by the convolution and fully connected layers.
          name: A string representing the name of the network.

        Raises:
          ValueError: If `input_tensor_spec` contains more than one observation. Or
            if `action_spec` contains more than one action.
        """
        validate_specs(action_specs, input_tensor_specs)

        super(MixedQNetwork, self).__init__(
            input_tensor_spec=input_tensor_specs,
            state_spec=(),
            name=name,
            mask_split_fn=mask_split_fn)

        self._input_tensor_specs = input_tensor_specs
        self._action_specs = action_specs
        self._discrete_action_key, self._continuous_action_key = 'discrete', 'continuous'

        self._num_filters_per_block = [16, 16]
        self._blocks = [1, 1]

        self._spatial_names = spatial_names
        self._structured_names = structured_names
        self._embeddings = None
        self._previous_action_embeddings = None

        self._previous_action_key = 'previous_action'
        self._previous_action_spec = input_tensor_specs[self._previous_action_key] if self._previous_action_key in \
                                                  input_tensor_specs else None

        self._create_embedding()

        spatial_encoders = dict()
        structured_encoders = dict()
        spatial_num_actions = 0
        structured_num_actions = 0

        # build encoders
        for type_name in self._spatial_names:
            num_actions = 0
            if type_name in self._action_specs:
                action_spec = self._action_specs[type_name]
                num_actions = action_spec.maximum[0] - action_spec.minimum[0] + 1
                spatial_num_actions += num_actions

            if type_name in self._input_tensor_specs:

                # infer spatial encoder input spec
                raw_encoder_input_tensor_specs = self._input_tensor_specs[type_name]
                first_spec = raw_encoder_input_tensor_specs[0]
                encoder_input_tensor_spec_num_channels = 0
                for spec in raw_encoder_input_tensor_specs:
                    if spec.name in self._embeddings[type_name]:
                        encoder_input_tensor_spec_num_channels += self._embeddings[type_name][spec.name].shape[1]
                    else:
                        encoder_input_tensor_spec_num_channels += spec.shape[0]
                # need to convert raw_encoder_input_tensor_specs by using embeddings
                encoder_input_tensor_spec_shape = list(first_spec.shape)+[encoder_input_tensor_spec_num_channels]
                encoder_input_tensor_spec = tensor_spec.BoundedTensorSpec(
                    shape=tuple(encoder_input_tensor_spec_shape), dtype=tf.float32,
                    name='encoder_input_tensor_spec_'+type_name,
                    minimum=(-np.inf,), maximum=(np.inf,))

                if mask_split_fn:
                    encoder_input_tensor_spec, _ = mask_split_fn(encoder_input_tensor_spec)

                spatial_encoder = shallow_resnet.ShallowResnet(
                    encoder_input_tensor_spec,
                    num_classes=num_actions,
                    blocks=self._blocks,
                    num_filters_per_block=self._num_filters_per_block,
                    dtype='float32',
                    batch_size=None,
                    use_l2_regularizer=True,
                    softmax=True,
                    name=type_name)
                spatial_encoders[type_name] = spatial_encoder

        for type_name in self._structured_names:
            num_actions = 0
            if type_name in self._action_specs:
                action_spec = self._action_specs[type_name]
                num_actions = action_spec.maximum[0] - action_spec.minimum[0] + 1
                structured_num_actions += num_actions

            if type_name in self._input_tensor_specs:
                encoder_input_tensor_spec = self._input_tensor_specs[type_name]

                if mask_split_fn:
                    encoder_input_tensor_spec, _ = mask_split_fn(encoder_input_tensor_spec)

                structured_encoder = encoding_network.EncodingNetwork(
                    encoder_input_tensor_spec,
                    preprocessing_layers=preprocessing_layers,
                    preprocessing_combiner=preprocessing_combiner,
                    conv_layer_params=conv_layer_params,
                    fc_layer_params=fc_layer_params,
                    dropout_layer_params=dropout_layer_params,
                    activation_fn=activation_fn,
                    kernel_initializer=kernel_initializer,
                    batch_squash=batch_squash,
                    dtype=dtype)
                structured_encoders[type_name] = structured_encoder

        # build q value layers
        spatial_q_value_layer = dict()
        for type_name in self._spatial_names:
            if type_name in self._action_specs:
                action_spec = self._action_specs[type_name]
                spatial_num_actions = action_spec.maximum[0] - action_spec.minimum[0] + 1
                if spatial_num_actions > 0:
                    spatial_q_value_layer[type_name] = tf.keras.layers.Conv2D(
                        filters=spatial_num_actions,
                        kernel_size=1,
                        strides=1,
                        activation=activation_fn,
                        kernel_initializer=kernel_initializer,
                        use_bias=True,
                        dtype=dtype,
                        name='%s/conv2d' % 'spatial_q_'+type_name)

        structured_q_value_layer = dict()
        for type_name in self._structured_names:
            if type_name in self._action_specs:
                action_spec = self._action_specs[type_name]
                structured_num_actions = action_spec.maximum[0] - action_spec.minimum[0] + 1
                if structured_num_actions > 0:
                    structured_q_value_layer[type_name] = tf.keras.layers.Dense(
                        structured_num_actions,
                        activation=None,
                        kernel_initializer=tf.compat.v1.initializers.random_uniform(
                            minval=-0.03, maxval=0.03),
                        bias_initializer=tf.compat.v1.initializers.constant(-0.2),
                        dtype=dtype)

        self._spatial_encoders = spatial_encoders
        self._structured_encoders = structured_encoders
        self._spatial_q_value_layer = spatial_q_value_layer
        self._structured_q_value_layer = structured_q_value_layer

    def _create_embedding(self):

        embeddings = dict()
        obs_type_names = list(self._spatial_names) + list(self._structured_names)
        for type_name in obs_type_names:
            if type_name in self._input_tensor_specs:
                spatial_tensor_specs = self._input_tensor_specs[type_name]
                embeddings[type_name] = dict()
                for spec in spatial_tensor_specs:
                    if spec.dtype in (tf.int32, tf.int64):
                        ##TODO: implement flexible specification of embedding size
                        obs_embedding_size = 4
                        state_num_classes = spec.maximum[0] - spec.minimum[0] + 1
                        obs_embeddings = tf.Variable(
                            tf.random.normal([state_num_classes, obs_embedding_size]), name="obs_embeddings_" + spec.name)
                        embeddings[type_name][spec.name] = obs_embeddings

        if self._previous_action_spec is not None:
            action_embedding_size = 8
            self._discrete_action_key = "discrete"
            assert isinstance(self._previous_action_spec, dict)

            # only need to create embedding for discrete actions
            if self._discrete_action_key in self._previous_action_spec:
                action_num_classes = sum([spec.maximum[0]-spec.minimum[0]+1
                               for spec in tf.nest.flatten(self._previous_action_spec[self._discrete_action_key])])
                # to allow for masked actions
                action_num_classes += 1
                self._previous_action_embeddings = tf.Variable(
                    tf.random.normal([action_num_classes, action_embedding_size]), name="action_embeddings")

        self._embeddings = embeddings

        return

    def call(self, observation, step_type=None, network_state=()):
        """

        :param observation: A dict of list of observations, including screen, minimaps, structured
          input. All of them should be batched. Since they are unstacked to seperate different types of
          categorical data, they are all of shape [batch_size, height, width].
        action: action taken in previous step, including action type and action arguments.
        :param step_type:
        :param network_state:
        :return:
        """
        previous_action = observation[self._previous_action_key] if self._previous_action_key in observation else None

        mask_split_fn = self.mask_split_fn

        if mask_split_fn:
            # Extract the network-specific portion of the observation.
            observation, _ = mask_split_fn(observation)

        if not isinstance(observation, dict):
            raise ValueError('observation for mixed q network should be dict')

        obs_type_names = list(self._spatial_names) + list(self._structured_names)

        # embed spatial categorical features and combine embeded features with continuous features
        spatial_states = dict()
        structured_states = []
        for type_name in self._spatial_names:
            assert type_name in self._embeddings, \
                "obs {} should be included in embedding".format(type_name)
            states = []
            spatial_tensor_specs = self._input_tensor_specs[type_name]
            for i, spec in enumerate(spatial_tensor_specs):
                if spec.dtype in (tf.int32, tf.int64):
                    assert spec.name in self._embeddings[type_name], \
                        "obs {} should be included in embedding[{}]".format(spec.name, type_name)
                    state = observation[type_name][i]
                    state_embed = tf.nn.embedding_lookup(self._embeddings[type_name][spec.name], state)
                else:
                    # embedding expands feature dim by 1.
                    state_embed = tf.cast(tf.expand_dims(observation[type_name][i], axis=-1), dtype=spec.dtype)
                states.append(state_embed)
            states = tf.concat(states, axis=-1)
            spatial_states[type_name] = states

        # embed previous action states
        # implemented inclusion of continuous arguments from previous action, such as screen/minimap coordinates
        action_states = None
        if previous_action is not None:
            assert len(previous_action) > 0
            assert isinstance(previous_action, dict)
            assert isinstance(self._previous_action_spec, dict)
            tf.nest.assert_same_structure(self._previous_action_spec, previous_action)

            action_states = []
            if self._discrete_action_key in previous_action:
                if previous_action[self._discrete_action_key].shape.ndims == 2:
                    assert previous_action[self._discrete_action_key].shape[0] == 1
                    previous_action[self._discrete_action_key] = tf.squeeze(previous_action[self._discrete_action_key],
                                                                            axis=-1)
                action_states.append(tf.nn.embedding_lookup(
                            self._previous_action_embeddings, previous_action[self._discrete_action_key]))
            if self._continuous_action_key in previous_action:
                assert previous_action[self._continuous_action_key].shape.ndims == 2
                action_states.append(previous_action[self._continuous_action_key])
            action_states = tf.concat(action_states, axis=-1)

        # embed structured states.
        for type_name in self._structured_names:
            if type_name in self._input_tensor_specs:
                assert type_name in self._embeddings, \
                    "obs {} should be included in embedding".format(type_name)
                ##TODO: how to handle non-categorical data in structured data.
                structured_tensor_specs = self._input_tensor_specs[type_name]
                for i, spec in enumerate(structured_tensor_specs):
                    # embed discrete actions and concatenate continuous actions
                    if spec.dtype in (tf.int32, tf.int64):
                        assert spec.name in self._embeddings[type_name], \
                            "obs {} should be included in embedding[{}]".format(spec.name, type_name)
                        state = observation[type_name][i]
                        state_embed = tf.nn.embedding_lookup(self._embeddings[type_name][spec.name], state)
                    else:
                        state_embed = tf.expand_dims(observation[type_name][i], axis=-1)
                    structured_states.append(state_embed)
        if len(structured_states) > 0:
            structured_states = tf.concat(structured_states, axis=-1)

        ##TODO: only screen data is processed here, need to consider how to incorporate minimap data.
        spatial_info_to_implement = ['screen', 'minimap']
        structured_info_to_implement = []
        spatial_features = dict()
        # feed spatial data through spatial encoders
        for type_name in spatial_info_to_implement:
            if type_name in self._spatial_encoders:
                spatial_features[type_name] = self._spatial_encoders[type_name](spatial_states[type_name])

        # concatenate screen and minimap features with each other
        if len(spatial_info_to_implement) == 2:
            reduced_screen_features = tf.reduce_mean(spatial_features['screen'], axis=[1, 2], keepdims=True)
            reduced_minimap_features = tf.reduce_mean(spatial_features['minimap'], axis=[1, 2], keepdims=True)
            screen_shape = spatial_features['screen'].shape[1]
            minimap_shape = spatial_features['minimap'].shape[1]

            broadcast_screen2minimap_features = tf.tile(reduced_screen_features, [1, minimap_shape, minimap_shape, 1])
            broadcast_minimap2screen_features = tf.tile(reduced_minimap_features, [1, screen_shape, screen_shape, 1])
            spatial_features['screen'] = tf.concat([spatial_features['screen'], broadcast_minimap2screen_features],
                                                    axis=-1)
            spatial_features['minimap'] = tf.concat([spatial_features['minimap'], broadcast_screen2minimap_features],
                                                    axis=-1)

        # feed structured data through structured encoders and concatenate with spatial data
        if isinstance(structured_states, tf.Tensor) and len(structured_info_to_implement) > 0:
            structured_states = self._structured_encoders[structured_info_to_implement[0]](structured_states)
            assert len(structured_states.shape) == 2
            structured_states = tf.expand_dims(tf.expand_dims(structured_states, axis=1), axis=1)
            for k, v in spatial_features.items():
                broadcast_structured_states = tf.tile(structured_states,
                                            [1, v.shape[1], v.shape[2], 1])
                spatial_features[k] = tf.concat([v, broadcast_structured_states], axis=-1)

        # concatenate embeded action states with spatial states.
        if action_states is not None:
            action_states = tf.expand_dims(tf.expand_dims(action_states, axis=1), axis=1)
            for k, v in spatial_features.items():
                broadcast_action_states = tf.tile(action_states, [1, v.shape[1], v.shape[2], 1])
                spatial_features[k] = tf.concat([v, broadcast_action_states], axis=-1)

        spatial_q_values, structured_q_values = dict(), dict()
        if self._spatial_q_value_layer is not None:
            for k, v in spatial_features.items():
                if k in self._spatial_q_value_layer:
                    spatial_q_values[k] = self._spatial_q_value_layer[k](v)

        combined_features = []
        for k, v in spatial_features.items():
            combined_features.append(tf.reduce_mean(v, axis=[1, 2]))
        combined_features = tf.concat(combined_features, axis=-1)
        for name, layer in self._structured_q_value_layer.items():
            structured_q_values[name] = layer(combined_features)

        return (spatial_q_values, structured_q_values), network_state
