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

"""Policy implementation that generates actions squentially from multiple policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.distributions import shifted_categorical

import copy
from tensorflow.python.framework import tensor_spec
import numpy as np

# TODO(b/131405384): Remove this once Deterministic does casting internally.
class DeterministicWithLogProb(tfp.distributions.Deterministic):
    """Thin wrapper around Deterministic that supports taking log_prob."""

    def _log_prob(self, x):
        """Takes log-probs by casting to tf.float32 instead of self.dtype."""
        return tf.math.log(tf.cast(self.prob(x), dtype=tf.float32))


class SequentialPolicy(tf_policy.Base):
    """Returns greedy samples of a given policy."""

    def __init__(self, policies, name=None):
        """Builds a greedy TFPolicy wrapping the given policy.

        Args:
          policy: A policy implementing the tf_policy.Base interface.
          name: The name of this policy. All variables in this module will fall
            under that name. Defaults to the class name.
        """

        if not isinstance(policies, (list, tuple)):
            policies = [policies]
        self._time_step_spec = [p.time_step_spec for p in policies]
        self._action_spec = [p.action_spec for p in policies]
        self._policy_state_spec = [p.policy_state_spec for p in policies]
        self._info_spec = [p.info_spec for p in policies]
        self._policy = policies
        self._num_specs = len(self._time_step_spec)

        self._emit_log_probability = [p.emit_log_probability for p in policies]
        self._raw_action_key, self._transformed_action_key = 'raw', 'transformed'
        self._transform_specs()

        super(SequentialPolicy, self).__init__(
            self._time_step_spec,
            self._action_spec,
            self._policy_state_spec,
            self._info_spec,
            emit_log_probability=self._emit_log_probability[0],
            clip=False,
            name=name)
        self._wrapped_policy = policies

    def _variables(self):
        return [p.variables() for p in self._wrapped_policy]

    def _transform_specs(self):

        # transform specs for replay_buffer and policy base class

        # only the first time_step_spec is needed as no previous_action is needed
        self._time_step_spec = self._time_step_spec[0]

        # the discrete action taken at each n step is need to compute q values,
        # so action spec is designed to take shape of num_specs
        num_actions = [int(1e5) for i in range(self._num_specs)]
        assert len(num_actions) == self._num_specs

        self._action_spec = dict()
        self._action_spec[self._raw_action_key] = tensor_spec.BoundedTensorSpec(
                    shape=(self._num_specs, ), dtype=tf.int32,
                    name=self._raw_action_key+'action',
                    minimum=tuple([0]*self._num_specs),
                    maximum=tuple(num_actions))
        # transformed action is set to dtype float32 due to transformed coordinates
        num_transformed_actions = 3
        self._action_spec[self._transformed_action_key] = tensor_spec.BoundedTensorSpec(
                    shape=(self._num_specs, num_transformed_actions), dtype=tf.float32,
                    name=self._transformed_action_key+'action',
                    minimum=(0, ),
                    maximum=(max(num_actions), ))

        self._policy_state_spec = self._policy_state_spec[0]
        self._info_spec = self._info_spec[0]

        return


    def _feedforward(self, time_step, policy_state, seed):
        """Feedforward network pass.

         Args:
           time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
           policy_state: A Tensor, or a nested dict, list or tuple of Tensors
             representing the previous policy_state.
           seed: Seed to use if action performs sampling (optional).

         Returns:
           A `PolicyStep` named tuple containing:
             `action`: An action Tensor matching the `action_spec()`.
             `state`: A policy state tensor to be fed into the next call to action.
             `info`: Optional side information such as action log probabilities.
         """

        previous_action_key = 'previous_action'
        func_action_key = 'func_action'
        available_actions_key = 'available_actions'

        assert isinstance(time_step.observation, dict)
        previous_action = None
        if previous_action_key in time_step.observation:
            time_step.observation[previous_action_key] = previous_action

        discrete_action_key, continuous_action_key = 'discrete', 'continuous'
        action_transform_key = 'inverse_transform_action'

        raw_action = []
        transformed_actions = []
        action_logits = []

        num_wrapped_policies = len(self._wrapped_policy)
        time_step = tf.nest.map_structure(lambda x: x, time_step)

        for i, policy in enumerate(self._wrapped_policy):
            if i == 0:
                assert available_actions_key in time_step.observation
            elif available_actions_key in time_step.observation:
                time_step.observation.pop(available_actions_key)

            action_step, distribution_step = policy.action_distribution(time_step, policy_state, seed)

            raw_action.append(action_step.action)
            # each action has different action logit size so we can't concat logits
            action_logits.append(distribution_step.action.logits)

            if action_transform_key in type(policy).__dict__.keys():
                action = policy.inverse_transform_action(action_step.action)
            else:
                assert 'wrapped_policy' in type(policy).__dict__.keys()
                assert action_transform_key in type(policy.wrapped_policy).__dict__.keys()
                action = policy.wrapped_policy.inverse_transform_action(action_step.action)

            transformed_actions.append(tf.expand_dims(action, axis=1))

            # create function action tensor
            if i == 0:
                time_step.observation[func_action_key] = dict()
                time_step.observation[func_action_key][discrete_action_key] = tf.cast(action[:, 0], tf.int32)

            # create previous action tensor
            time_step.observation[previous_action_key] = dict()
            if i < num_wrapped_policies - 1 and \
                    discrete_action_key in self._wrapped_policy[i+1].time_step_spec.observation['previous_action']:
                time_step.observation[previous_action_key][discrete_action_key] = tf.cast(action[:, 0], tf.int32)
            if i < num_wrapped_policies - 1 and \
                    continuous_action_key in self._wrapped_policy[i+1].time_step_spec.observation['previous_action']:
                # normalize continuous parameters in HeteroQPolicy
                time_step.observation[previous_action_key][continuous_action_key] = action[:, 1:]

        raw_action = tf.concat(raw_action, axis=-1)
        transformed_actions = tf.concat(transformed_actions, axis=1)

        return raw_action, transformed_actions, action_logits

    def _action(self, time_step, policy_state, seed):

        """Implementation of `action`.

         Args:
           time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
           policy_state: A Tensor, or a nested dict, list or tuple of Tensors
             representing the previous policy_state.
           seed: Seed to use if action performs sampling (optional).

         Returns:
           A `PolicyStep` named tuple containing:
             `action`: An action Tensor matching the `action_spec()`.
             `state`: A policy state tensor to be fed into the next call to action.
             `info`: Optional side information such as action log probabilities.
         """

        raw_action, transformed_actions, _ = self._feedforward(time_step, policy_state, seed)

        def _to_distribution(action_or_distribution):
            if isinstance(action_or_distribution, tf.Tensor):
                # This is an action tensor, so wrap it in a deterministic distribution.
                return tfp.distributions.Deterministic(loc=action_or_distribution)
            return action_or_distribution

        combined_actions = dict()
        combined_actions[self._raw_action_key] = raw_action
        combined_actions[self._transformed_action_key] = transformed_actions
        distributions = tf.nest.map_structure(_to_distribution,
                                              combined_actions)
        distribution_step = policy_step.PolicyStep(distributions, policy_state)

        seed_stream = tfp.distributions.SeedStream(seed=seed, salt='sc2_sequential_policy')
        actions = tf.nest.map_structure(lambda d: d.sample(seed=seed_stream()),
                                        distribution_step.action)
        info = distribution_step.info

        return distribution_step._replace(action=actions, info=info)

    def _action_distribution(self, time_step, policy_state, seed):
        """Implementation of `_action_distribution`.

         Args:
           time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
           policy_state: A Tensor, or a nested dict, list or tuple of Tensors
             representing the previous policy_state.
           seed: Seed to use if action performs sampling (optional).

         Returns:
           A `PolicyStep` named tuple containing:
             `action`: An action Tensor matching the `action_spec()`.
             `state`: A policy state tensor to be fed into the next call to action.
             `info`: Optional side information such as action log probabilities.
           A `PolicyStep` named tuple containing:
             `action`: A Logit Tensor representing Q values`.
             `state`: A policy state tensor to be fed into the next call to action.
             `info`: Optional side information such as action log probabilities.
         """

        raw_action, transformed_actions, action_logits = self._feedforward(time_step, policy_state, seed)

        def _to_distribution(action_or_distribution):
            if isinstance(action_or_distribution, tf.Tensor):
                # This is an action tensor, so wrap it in a deterministic distribution.
                return tfp.distributions.Deterministic(loc=action_or_distribution)
            return action_or_distribution

        combined_actions = dict()
        combined_actions[self._raw_action_key] = raw_action
        combined_actions[self._transformed_action_key] = transformed_actions

        q_distributions = [shifted_categorical.ShiftedCategorical(
            logits=logit, dtype=tf.int32, shift=0) for logit in action_logits]
        distribution_steps = [policy_step.PolicyStep(q_dist, policy_state) for q_dist in q_distributions]

        actions = tf.nest.map_structure(_to_distribution,
                                              combined_actions)
        action_step = policy_step.PolicyStep(actions, policy_state)
        seed_stream = tfp.distributions.SeedStream(seed=seed, salt='sc2_sequential_policy')
        actions = tf.nest.map_structure(lambda d: d.sample(seed=seed_stream()),
                                        action_step.action)

        return action_step._replace(action=actions), distribution_steps


    def _distribution(self, time_step, policy_state):
        raise NotImplementedError





