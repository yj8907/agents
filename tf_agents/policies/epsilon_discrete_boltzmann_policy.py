#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:03:36 2019

@author: yangjiang
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies import discrete_boltzmann_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.utils import nest_utils

tfd = tfp.distributions


class EpsilonDiscreteBoltzmannPolicy(tf_policy.Base):
  """Returns epsilon-greedy samples of a given policy."""

  def __init__(self, policy, epsilon, env_action_spec, name=None):
    """Builds an epsilon-greedy MixturePolicy wrapping the given policy.

    Args:
      policy: A policy implementing the tf_policy.Base interface.
      epsilon: The probability of taking the random action represented as a
        float scalar, a scalar Tensor of shape=(), or a callable that returns a
        float scalar or Tensor.
      name: The name of this policy.

    Raises:
      ValueError: If epsilon is invalid.
    """
    self._discrete_boltzmann_policy = discrete_boltzmann_policy.DiscreteBoltzmannPolicy(
          policy, temperature=1.0)
    self._epsilon = epsilon
    self._env_action_spec = env_action_spec
    
    self._random_policy = random_tf_policy.RandomTFPolicy(
        policy.time_step_spec,
        env_action_spec,
        emit_log_probability=policy.emit_log_probability)
    super(EpsilonDiscreteBoltzmannPolicy, self).__init__(
        policy.time_step_spec,
        policy.action_spec,
        policy.policy_state_spec,
        policy.info_spec,
        emit_log_probability=policy.emit_log_probability,
        name=name)

  def _variables(self):
    return self._greedy_policy.variables()

  def _get_epsilon(self):
    if callable(self._epsilon):
      return self._epsilon()
    else:
      return self._epsilon

  def _action(self, time_step, policy_state, seed):
    seed_stream = tfd.SeedStream(seed=seed, salt='epsilon_greedy')
    discrete_boltzmann_action = self._discrete_boltzmann_policy.action(time_step, policy_state)
    random_action = self._random_policy.action(time_step, (), seed_stream())

    outer_shape = nest_utils.get_outer_shape(time_step, self._time_step_spec)
    rng = tf.random.uniform(
        outer_shape, maxval=1.0, seed=seed_stream(), name='epsilon_rng')
    cond = tf.greater(rng, self._get_epsilon())

    # Selects the action/info from the random policy with probability epsilon.
    # TODO(b/133175894): tf.compat.v1.where only supports a condition which is
    # either a scalar or a vector. Use tf.compat.v2 so that it can support any
    # condition whose leading dimensions are the same as the other operands of
    # tf.where.
    outer_ndims = int(outer_shape.shape[0])
    if outer_ndims >= 2:
      raise ValueError(
          'Only supports batched time steps with a single batch dimension')
      
    # provide logits for random action
    random_action_logits = tf.one_hot(random_action.action, 
                      depth=self._env_action_spec.maximum-self._env_action_spec.minimum+1)
    random_action_wlogits = tf.concat([tf.cast(tf.expand_dims(random_action.action, axis=1), tf.float32),
                                      random_action_logits], axis=-1)
    
#    action = tf.compat.v1.where(cond, discrete_boltzmann_action.action,
#                                random_action_wlogits)
    action = discrete_boltzmann_action.action
    print('action')
    print(action)
    print(random_action_wlogits)

    if discrete_boltzmann_action.info:
      if not random_action.info:
        raise ValueError('Incompatible info field')
      info = tf.compat.v1.where(cond, discrete_boltzmann_action.info, random_action.info)
    else:
      if random_action.info:
        raise ValueError('Incompatible info field')
      info = ()

    # The state of the epsilon greedy policy is the state of the underlying
    # greedy policy (the random policy carries no state).
    # It is commonly assumed that the new policy state only depends only
    # on the previous state and "time_step", the action (be it the greedy one
    # or the random one) does not influence the new policy state.
    state = discrete_boltzmann_action.state

    return policy_step.PolicyStep(action, state, info)

  def _distribution(self, time_step, policy_state):
    raise NotImplementedError(
        'EpsilonGreedyPolicy does not support distributions yet.')
