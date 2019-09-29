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

"""Tests for tf_agents.agents.td3.td3_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.env import mock_sc2_env
import tensorflow as tf
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
import tf_agents.policies.hetero_q_policy as hetero_q_policy
from tf_agents.agents.td3 import td3_dqn_agent
from tf_agents.networks import mixed_q_network

from absl.testing import parameterized

import tensorflow_probability as tfp
from tf_agents.policies import greedy_policy
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils

spatial_keys = ["minimap", "screen"]
structured_keys = ['structured']

screen_size = 24
minimap_size = 16


def extract_action_spec(step, act_wrapper):
    action_specs = [act_wrapper._merged_spatial_action_spec,
                    act_wrapper._merged_structured_action_spec]

    if step < -1:
        print("generate None spec for step < -1")
        return None

    action_spec = dict()
    if step >= 0:
        for spec in action_specs:
            for k, v in spec.items():
                if step < len(v):
                    action_spec[k] = v[step]
    else:
        action_spec['structured'] = tensor_spec.BoundedTensorSpec(
            shape=(1,), dtype=np.int32, name="sc2_func_action_spec",
            minimum=(0,),
            maximum=(len(act_wrapper.func_ids) - 1,))

    return action_spec


def extract_previous_action_spec(current_action_step, act_wrapper):
    previous_action_spec = None
    if current_action_step == -1:
        previous_action_spec = None
    elif current_action_step == 0:
        previous_action_spec = dict()
        previous_action_spec['discrete'] = tensor_spec.BoundedTensorSpec(
            shape=(1,), dtype=np.int32,
            name='functions',
            minimum=(0,), maximum=(len(act_wrapper.func_ids) - 1,))
    elif current_action_step == 1:
        action_specs = extract_action_spec(current_action_step - 1, act_wrapper)
        num_discrete_actions = sum([v.maximum[0] - v.minimum[0] + 1
                                    for k, v in action_specs.items()])

        previous_action_spec = dict()
        previous_action_spec['discrete'] = tensor_spec.BoundedTensorSpec(
            shape=(1,), dtype=np.int32,
            name='discrete_func',
            minimum=(0,), maximum=(num_discrete_actions - 1,))
        previous_action_spec['continuous'] = tensor_spec.BoundedTensorSpec(
            shape=(2,), dtype=np.float32,
            name='continuous_func',
            minimum=(-np.inf,), maximum=(np.inf,))

    return previous_action_spec


def act_obs_wrapper_generator():
    mock_env = mock_sc2_env.SC2TestEnv(map_name=0, agent_interface_format=[
        features.parse_agent_interface_format(feature_screen=screen_size, feature_minimap=minimap_size)])

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

    return obs_wrapper, act_wrapper


def arg_spec_generator(action_step):
    obs_wrapper, act_wrapper = act_obs_wrapper_generator()

    action_specs = \
        extract_action_spec(action_step, act_wrapper)

    input_tensor_specs = obs_wrapper.obs_spec
    input_tensor_specs['previous_action'] = extract_previous_action_spec(action_step, act_wrapper)

    if action_step >= 0:
        func_action_mask = act_wrapper.func_action_mask[action_step]
    else:
        func_action_mask = None

    return input_tensor_specs, action_specs, func_action_mask


def mixed_q_network_generator(action_steps=None):
    input_tensor_specs = []
    action_specs = []
    q_networks = []
    func_masks = []

    if action_steps is None:
        action_steps = [-1, 0, 1]
    else:
        assert isinstance(action_steps, (tuple, list))

    for action_step in action_steps:
        with tf.compat.v1.variable_scope('step' + str(action_step + 1), reuse=tf.compat.v1.AUTO_REUSE) as scope:
            input_spec, action_spec, func_action_mask = arg_spec_generator(action_step)
            q_net = mixed_q_network.MixedQNetwork(input_spec, action_spec,
                                                  name='MixedQNetwork' + str(action_step + 1))
            input_tensor_specs.append(input_spec)
            action_specs.append(action_spec)
            q_networks.append(q_net)
            func_masks.append(func_action_mask)

    return input_tensor_specs, action_specs, q_networks, func_masks


def observation_generator(action_step, action_spec):
    input_tensor_specs, action_specs, func_action_mask = arg_spec_generator(action_step)
    screen_image = tf.ones((1, screen_size, screen_size, len(input_tensor_specs['screen'])), tf.float32)
    minimap_image = tf.ones((1, minimap_size, minimap_size, len(input_tensor_specs['minimap'])), tf.float32)

    observation = dict()
    observation['screen'] = tf.unstack(screen_image, len(input_tensor_specs['screen']), axis=-1)
    observation['minimap'] = tf.unstack(minimap_image, len(input_tensor_specs['minimap']), axis=-1)

    for i in range(len(input_tensor_specs['screen'])):
        observation['screen'][i] = tf.cast(observation['screen'][i],
                                           input_tensor_specs['screen'][i].dtype)
    for i in range(len(input_tensor_specs['minimap'])):
        observation['minimap'][i] = tf.cast(observation['minimap'][i],
                                            input_tensor_specs['minimap'][i].dtype)

    previous_action = None
    if action_step >= 0:
        previous_action = dict()
        for k, v in action_spec.items():
            if k == 'discrete':
                previous_action[k] = tf.ones((1,), np.int32) * 1
            if k == 'continuous':
                previous_action[k] = tf.ones((1, v.shape[0]), np.float32) * 1

    return observation, previous_action


class TD3DqnAgentTest(test_utils.TestCase):

    def setUp(self):
        super(TD3DqnAgentTest, self).setUp()

        self._obs_specs, self._action_specs, self._q_networks, self._masks = \
            mixed_q_network_generator()

        from tf_agents.trajectories import time_step as ts
        use_previous_actions = [False, True, True]

    def testCreateAgent(self):
        time_step_specs = []
        for step in range(3):
            time_step_specs.append(ts.time_step_spec(self._obs_specs[step]))

            # prev_act_spec = self._obs_specs[step]['previous_action']
            # observations, prev_act = observation_generator(step - 1, prev_act_spec)
            # observations['previous_action'] = prev_act
            # time_step = ts.restart(observations, batch_size=1)
        critic_learning_rate = 1e-3
        critic_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate)

        self._agent = td3_dqn_agent.Td3DqnAgent(time_step_spec=time_step_specs, action_spec=self._action_specs,
                            q_networks=self._q_networks, critic_optimizer=critic_optimizer,
                            boltzmann_temperature=10, action_params_mask=self._masks,
                            name='sc2_agent')

    def testCriticLoss(self):
        # The loss is now 119.3098526. Investigate this.
        # self.skipTest('b/123772477')
        self.setUp()
        self.testCreateAgent()

        observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
        time_steps = ts.restart(observations, batch_size=2)
        actions = [tf.constant([[5], [6]], dtype=tf.float32)]

        rewards = tf.constant([10, 20], dtype=tf.float32)
        discounts = tf.constant([0.9, 0.9], dtype=tf.float32)
        next_observations = [tf.constant([[5, 6], [7, 8]], dtype=tf.float32)]
        next_time_steps = ts.transition(next_observations, rewards, discounts)

        # TODO(b/123772477): The loss changed from 119.054 to 118.910903931.
        expected_loss = 118.9109
        loss = self._agent.critic_loss(time_steps, actions, next_time_steps)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        loss_ = self.evaluate(loss)
        self.assertAllClose(loss_, expected_loss)

    def testActorLoss(self):
        agent = td3_agent.Td3Agent(
            self._time_step_spec,
            self._action_spec,
            critic_network=self._critic_net,
            actor_network=self._unbounded_actor_net,
            actor_optimizer=None,
            critic_optimizer=None)

        observations = [tf.constant([[1, 2], [3, 4]], dtype=tf.float32)]
        time_steps = ts.restart(observations, batch_size=2)

        expected_loss = 4.0
        loss = agent.actor_loss(time_steps)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        loss_ = self.evaluate(loss)
        self.assertAllClose(loss_, expected_loss)

    def testPolicyProducesBoundedAction(self):
        agent = td3_agent.Td3Agent(
            self._time_step_spec,
            self._action_spec,
            critic_network=self._critic_net,
            actor_network=self._bounded_actor_net,
            actor_optimizer=None,
            critic_optimizer=None)

        observations = [tf.constant([[1, 2]], dtype=tf.float32)]
        time_steps = ts.restart(observations, batch_size=1)
        action = agent.policy.action(time_steps).action[0]
        self.assertEqual(action.shape.as_list(), [1, 1])

        self.evaluate([
            tf.compat.v1.global_variables_initializer(),
            tf.compat.v1.local_variables_initializer()
        ])
        py_action = self.evaluate(action)
        self.assertTrue(all(py_action <= self._action_spec[0].maximum))
        self.assertTrue(all(py_action >= self._action_spec[0].minimum))

    def testPolicyAndCollectPolicyProducesDifferentActions(self):
        self.skipTest('b/125913845')

        agent = td3_agent.Td3Agent(
            self._time_step_spec,
            self._action_spec,
            critic_network=self._critic_net,
            actor_network=self._bounded_actor_net,
            actor_optimizer=None,
            critic_optimizer=None)

        observations = [tf.constant([[1, 2]], dtype=tf.float32)]
        time_steps = ts.restart(observations, batch_size=1)
        action = agent.policy.action(time_steps).action[0]
        collect_policy_action = agent.collect_policy.action(time_steps).action[0]
        self.assertEqual(action.shape, collect_policy_action.shape)

        self.evaluate([
            tf.compat.v1.global_variables_initializer(),
            tf.compat.v1.local_variables_initializer()
        ])
        py_action, py_collect_policy_action = self.evaluate(
            [action, collect_policy_action])
        self.assertNotEqual(py_action, py_collect_policy_action)


if __name__ == '__main__':
    tf.test.main()
