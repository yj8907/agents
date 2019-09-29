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
import tf_agents.policies.hetero_q_policy as hetero_q_policy

from absl.testing import parameterized

import tensorflow_probability as tfp
from tf_agents.policies import greedy_policy
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import test_utils

from tf_agents.agents.td3 import td3_dq_agent_test

class HeteroQPolicyTest(test_utils.TestCase):

    def setUp(self):
        super(HeteroQPolicyTest, self).setUp()
        self._obs_specs, self._action_specs, self._q_networks, self._masks = \
                td3_dq_agent_test.mixed_q_network_generator()
        self._time_step_spec = ts.time_step_spec(self._obs_spec)

    def testBuild(self, step=1):

        from tf_agents.trajectories import time_step as ts
        use_previous_actions = [False, True, True]

        policies = []
        for step in range(3):

            time_step_spec = ts.time_step_spec(self._obs_specs[step])
            action_spec = self._action_specs[step]
            q_net = self._q_networks[step]
            f_arg_mask = self._masks[step]
            use_previous = use_previous_actions[step]

            prev_act_spec = self._obs_specs[step]['previous_action']
            observations, prev_act = td3_dq_agent_test.observation_generator(step - 1, prev_act_spec)
            observations['previous_action'] = prev_act
            time_step = ts.restart(observations, batch_size=1)

            policy = hetero_q_policy.HeteroQPolicy(time_step_spec,
                           action_spec, mixed_q_network=q_net, func_arg_mask=f_arg_mask,
                           use_previous_action=use_previous_actions[step])
            policies.append(policy)

    def testMultipleActionsRaiseError(self):
        action_spec = [tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)] * 2
        with self.assertRaisesRegexp(
                NotImplementedError,
                'action_spec can only contain a single BoundedTensorSpec'):
            hetero_q_policy.HeteroQPolicy(
                self._time_step_spec, action_spec, q_network=DummyNet())

    def testAction(self):
        step = 0
        observations, previous_action, func_action_mask = observation_generator(step)
        time_step = ts.restart(observations, batch_size=1)

        policy = hetero_q_policy.HeteroQPolicy(
            self._time_step_spec, self._action_spec,
            mixed_q_network=self._mixed_q_network, func_arg_mask=func_action_mask)
        policy_step = policy._distribution(time_step, policy_state=())

        # action_step = policy.action(time_step, seed=1)

        tf.nest.assert_same_structure(policy_step, policy.policy_step_spec)
        self.assertEqual(action_step.action.shape.as_list(), [2, 1])
        self.assertEqual(action_step.action.dtype, tf.int32)
        # Initialize all variables
        self.evaluate(tf.compat.v1.global_variables_initializer())
        action = self.evaluate(action_step.action)
        self.assertTrue(np.all(action >= 0) and np.all(action <= 1))

    def testActionWithinBounds(self):
        bounded_action_spec = tensor_spec.BoundedTensorSpec([1],
                                                            tf.int32,
                                                            minimum=-6,
                                                            maximum=-5)
        policy = q_policy.QPolicy(
            self._time_step_spec, bounded_action_spec, q_network=DummyNet())

        observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        time_step = ts.restart(observations, batch_size=2)
        action_step = policy.action(time_step)
        self.assertEqual(action_step.action.shape.as_list(), [2, 1])
        self.assertEqual(action_step.action.dtype, tf.int32)
        # Initialize all variables
        self.evaluate(tf.compat.v1.global_variables_initializer())
        action = self.evaluate(action_step.action)
        self.assertTrue(np.all(action <= -5) and np.all(action >= -6))

    def testActionScalarSpec(self):
        action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 1)
        policy = q_policy.QPolicy(
            self._time_step_spec, action_spec, q_network=DummyNet())

        observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        time_step = ts.restart(observations, batch_size=2)
        action_step = policy.action(time_step, seed=1)
        self.assertEqual(action_step.action.shape.as_list(), [2])
        self.assertEqual(action_step.action.dtype, tf.int32)
        # Initialize all variables
        self.evaluate(tf.compat.v1.global_variables_initializer())
        action = self.evaluate(action_step.action)
        self.assertTrue(np.all(action >= 0) and np.all(action <= 1))

    def testActionList(self):
        action_spec = [tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 1)]
        policy = q_policy.QPolicy(
            self._time_step_spec, action_spec, q_network=DummyNet())
        observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        time_step = ts.restart(observations, batch_size=2)
        action_step = policy.action(time_step, seed=1)
        self.assertIsInstance(action_step.action, list)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        action = self.evaluate(action_step.action)
        self.assertLen(action, 1)
        # Extract contents from the outer list.
        action = action[0]
        self.assertTrue(np.all(action >= 0) and np.all(action <= 1))

    def testDistribution(self):
        policy = q_policy.QPolicy(
            self._time_step_spec, self._action_spec, q_network=DummyNet())

        observations = tf.constant([[1, 2]], dtype=tf.float32)
        time_step = ts.restart(observations, batch_size=1)
        distribution_step = policy.distribution(time_step)
        mode = distribution_step.action.mode()
        self.evaluate(tf.compat.v1.global_variables_initializer())
        # The weights of index 0 are all 1 and the weights of index 1 are all 1.5,
        # so the Q values of index 1 will be higher.
        self.assertAllEqual([[1]], self.evaluate(mode))

    def testUpdate(self):
        policy = q_policy.QPolicy(
            self._time_step_spec, self._action_spec, q_network=DummyNet())
        new_policy = q_policy.QPolicy(
            self._time_step_spec, self._action_spec, q_network=DummyNet())

        observations = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        time_step = ts.restart(observations, batch_size=2)

        self.assertEqual(policy.variables(), [])
        self.assertEqual(new_policy.variables(), [])

        action_step = policy.action(time_step, seed=1)
        new_action_step = new_policy.action(time_step, seed=1)

        self.assertEqual(len(policy.variables()), 2)
        self.assertEqual(len(new_policy.variables()), 2)
        self.assertEqual(action_step.action.shape, new_action_step.action.shape)
        self.assertEqual(action_step.action.dtype, new_action_step.action.dtype)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertEqual(self.evaluate(new_policy.update(policy)), None)

        action = self.evaluate(action_step.action)
        new_action = self.evaluate(new_action_step.action)
        self.assertTrue(np.all(action >= 0) and np.all(action <= 1))
        self.assertTrue(np.all(new_action >= 0) and np.all(new_action <= 1))
        self.assertAllEqual(action, new_action)

    def testActionSpecsCompatible(self):
        q_net = DummyNetWithActionSpec(self._action_spec)
        q_policy.QPolicy(self._time_step_spec, self._action_spec, q_net)

    def testActionSpecsIncompatible(self):
        network_action_spec = tensor_spec.BoundedTensorSpec([2], tf.int32, 0, 1)
        q_net = DummyNetWithActionSpec(network_action_spec)

        with self.assertRaisesRegexp(
                ValueError,
                'action_spec must be compatible with q_network.action_spec'):
            q_policy.QPolicy(self._time_step_spec, self._action_spec, q_net)

    def testMasking(self):
        batch_size = 1000
        num_state_dims = 5
        num_actions = 8
        observations = tf.random.uniform([batch_size, num_state_dims])
        time_step = ts.restart(observations, batch_size=batch_size)
        input_tensor_spec = tensor_spec.TensorSpec([num_state_dims], tf.float32)
        action_spec = tensor_spec.BoundedTensorSpec(
            [1], tf.int32, 0, num_actions - 1)

        mask = [0, 1, 0, 1, 0, 0, 1, 0]
        np_mask = np.array(mask)
        tf_mask = tf.constant([mask for _ in range(batch_size)])
        q_net = q_network.QNetwork(
            input_tensor_spec, action_spec,
            mask_split_fn=lambda observation: (observation, tf_mask))
        policy = q_policy.QPolicy(
            ts.time_step_spec(input_tensor_spec), action_spec, q_net)

        # Force creation of variables before global_variables_initializer.
        policy.variables()
        self.evaluate(tf.compat.v1.global_variables_initializer())

        # Sample from the policy 1000 times and ensure that invalid actions are
        # never chosen.
        action_step = policy.action(time_step)
        action = self.evaluate(action_step.action)
        self.assertEqual(action.shape, (batch_size, 1))
        self.assertAllEqual(np_mask[action], np.ones([batch_size, 1]))


