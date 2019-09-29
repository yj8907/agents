#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:15:44 2019

@author: yangjiang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.trajectories import policy_step
from tf_agents.policies import tf_policy

tfd = tfp.distributions


class DiscreteBoltzmannPolicyStep(policy_step.PolicyStep):
    
        def __new__(cls, policy, action_logits):
        
            self = super(DiscreteBoltzmannPolicyStep, cls).__new__(cls, policy.action, policy.state, policy.info)
            self.action_logits = action_logits
            
            return self


@gin.configurable
class DiscreteBoltzmannPolicy(tf_policy.Base):
  """Returns boltzmann samples of a given policy.

  The wrapped policy must expose a distribution parameterized by logits.
  """

  def __init__(self, policy, temperature=1.0, name=None):
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
    super(DiscreteBoltzmannPolicy, self).__init__(
        policy.time_step_spec,
        policy.action_spec,
        policy.policy_state_spec,
        policy.info_spec,
        emit_log_probability=policy.emit_log_probability,
        clip=False,
        name=name)
    self._temperature = temperature
    self._wrapped_policy = policy

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
    seed_stream = tfd.SeedStream(seed=seed, salt='ppo_policy')
    distribution_step = self._distribution(time_step, policy_state)
    
    action_distribution = distribution_step.action
    # distribution_step = DiscreteBoltzmannPolicyStep(distribution_step, action_distribution)        
    
    actions = tf.nest.map_structure(lambda d: d.sample(seed=seed_stream()),
                                    distribution_step.action)
    
    info = distribution_step.info
    if self.emit_log_probability:
      try:
        log_probability = tf.nest.map_structure(lambda a, d: d.log_prob(a),
                                                actions,
                                                distribution_step.action)
        info = policy_step.set_log_probability(info, log_probability)
      except:
        raise TypeError('%s does not support emitting log-probabilities.' %
                        type(self).__name__)
    
    actions = tf.concat([tf.cast(actions, tf.float32), tf.squeeze(action_distribution.probs, axis=1)], axis=-1)
    
    return distribution_step._replace(action=actions, info=info)



  def _distribution(self, time_step, policy_state):
    distribution_step = self._wrapped_policy.distribution(
        time_step, policy_state)
    if self._temperature is None:
      return distribution_step

    action_dist = tf.nest.map_structure(self._apply_temperature,
                                        distribution_step.action)
    return distribution_step._replace(action=action_dist)
