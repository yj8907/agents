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


from pysc2.env import mock_sc2_env
import numpy as np

import sys
import gin
import numpy as np
from absl import flags
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import protocol
from pysc2.env.environment import StepType

import tensorflow as tf
from tensorflow.python.framework import tensor_spec
import collections

from reaver.envs.tf_agent_sc2_v3 import ActionWrapper
from reaver.envs.tf_agent_sc2_v3 import ObservationWrapper

from tf_agents.networks.mixed_q_network import MixedQNetwork

def extract_action_spec(step, *action_specs):
    action_spec = dict()
    for spec in action_specs:
        for k, v in spec.items():
            if step < len(v):
                action_spec[k] = v[step]

    return action_spec

def mixed_q_netwwork_test():

    mock_env = mock_sc2_env.SC2TestEnv(map_name=0, agent_interface_format=[
        features.parse_agent_interface_format(feature_screen=16, feature_minimap=16)])

    obs_features = {
        'screen': ['player_relative', 'selected', 'visibility_map', 'unit_hit_points_ratio', 'unit_density'],
        'minimap': ['player_relative', 'selected', 'visibility_map', 'camera'],
        # available actions should always be present and in first position
        'non-spatial': ['available_actions', 'player']}

    action_ids = [0, 1, 2, 3, 4, 6, 7, 12, 13, 42, 44, 50, 91, 183, 234, 309, 331, 332, 333, 334, 451, 452, 490]
    act_wrapper = ActionWrapper(16, action_ids)
    act_wrapper.make_spec(mock_env.action_spec())

    obs_wrapper = ObservationWrapper(obs_features, action_ids)
    obs_wrapper.make_spec(mock_env.observation_spec())

    action_step = 1
    action_specs = \
        extract_action_spec(action_step, act_wrapper._merged_spatial_action_spec, act_wrapper._merged_structured_action_spec)
    pevious_action_specs = \
        extract_action_spec(action_step-1, act_wrapper._merged_spatial_action_spec, act_wrapper._merged_structured_action_spec)
    input_tensor_specs = obs_wrapper.obs_spec

    with tf.compat.v1.variable_scope('step2', reuse=tf.compat.v1.AUTO_REUSE) as scope:
        mixed_q_network = MixedQNetwork(input_tensor_specs, action_specs, pevious_action_specs)

    screen_image = tf.ones((1, 16, 16, len(input_tensor_specs['screen'])), tf.float32)
    minimap_image = tf.ones((1, 24, 24, len(input_tensor_specs['minimap'])), tf.float32)

    observation = dict()
    observation['screen'] = tf.unstack(screen_image, len(input_tensor_specs['screen']), axis=-1)
    observation['minimap'] = tf.unstack(minimap_image, len(input_tensor_specs['minimap']), axis=-1)

    for i in range(len(input_tensor_specs['screen'])):
        observation['screen'][i] = tf.cast(observation['screen'][i],
                                           input_tensor_specs['screen'][i].dtype)

    for i in range(len(input_tensor_specs['minimap'])):
        observation['minimap'][i] = tf.cast(observation['minimap'][i],
                                            input_tensor_specs['minimap'][i].dtype)

    previous_action = tf.ones((1), tf.int32)
    output = mixed_q_network(observation)

    return

