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

"""Policy implementation that applies temperature to a distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies import greedy_policy
from tf_agents.trajectories import policy_step
from tf_agents.policies import tf_policy
from tf_agents.utils import nest_utils

tfd = tfp.distributions

@gin.configurable
class EpsilonBoltzmannPolicy(tf_policy.Base):
    """Returns boltzmann samples of a given policy.

    The wrapped policy must expose a distribution parameterized by logits.
    """

    def __init__(self, policy, temperature=10.0, epsilon=0.1, remove_neg_inf=False, name=None):
        """Builds a BoltzmannPolicy wrapping the given policy.

        Args:
          policy: A policy implementing the tf_policy.Base interface, using
            a distribution parameterized by logits.
          temperature: Tensor or function that returns the temperature for sampling
            when `action` is called. This parameter applies when the action spec is
            discrete. If the temperature is close to 0.0 this is equivalent to
            calling `tf.argmax` on the output of the network.
          name: The name of this policy. All variables in this module will fall
            under that name. Defaults to the class name.
        """

        self._greedy_policy = greedy_policy.GreedyPolicy(policy, remove_neg_inf=remove_neg_inf)
        super(EpsilonBoltzmannPolicy, self).__init__(
            policy.time_step_spec,
            policy.action_spec,
            policy.policy_state_spec,
            policy.info_spec,
            emit_log_probability=policy.emit_log_probability,
            clip=False,
            name=name)
        self._temperature = temperature
        self._epsilon = epsilon
        self._wrapped_policy = policy

    @property
    def wrapped_policy(self):
        return self._wrapped_policy

    def _variables(self):
        return self._wrapped_policy.variables()

    def _get_temperature_value(self):
        if callable(self._temperature):
            return self._temperature()
        return self._temperature

    def _apply_temperature(self, dist):
        """Change the action distribution to incorporate the temperature."""
        logits = dist.logits / self._get_temperature_value()
        return dist.copy(logits=logits)

    def _get_epsilon(self):
        if callable(self._epsilon):
            return self._epsilon()
        else:
            return self._epsilon

    def _action(self, time_step, policy_state, seed):
        seed_stream = tfd.SeedStream(seed=seed, salt='epsilon_boltzmann')

        greedy_action, distribution_step = self._greedy_policy.action_distribution(
                        time_step, policy_state, seed_stream)

        action_dist = tf.nest.map_structure(self._apply_temperature,
                                            distribution_step.action)
        boltzmann_action = tf.nest.map_structure(lambda d: d.sample(seed=seed_stream()), action_dist)

        outer_shape = nest_utils.get_outer_shape(time_step, self._time_step_spec)
        rng = tf.random.uniform(
            outer_shape, maxval=1.0, seed=seed_stream(), name='epsilon_rng')
        cond = tf.greater(rng, self._get_epsilon())

        outer_ndims = int(outer_shape.shape[0])
        ##TODO: remove it to allow environment multiprocessing
        if outer_ndims >= 2:
            raise ValueError(
                'Only supports batched time steps with a single batch dimension')
        action = tf.compat.v1.where(cond, greedy_action.action, boltzmann_action)

        info = ()
        state = greedy_action.state

        return policy_step.PolicyStep(action, state, info)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError(
            'EpsilonGreedyPolicy does not support distributions yet.')