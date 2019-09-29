#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 01:40:27 2019

@author: yangjiang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.policies import boltzmann_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import greedy_policy

from tf_agents.networks.q_network import QNetwork

from tf_agents.agents import tf_agent
from tf_agents.policies import actor_policy
from tf_agents.policies import mixed_q_policy
from tf_agents.policies import gaussian_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import value_ops

class Td3Info(collections.namedtuple(
    'Td3Info', ('actor_loss', 'critic_loss'))):
  pass

def compute_td_targets(next_q_values, rewards, discounts):
  return tf.stop_gradient(rewards + discounts * next_q_values)

@gin.configurable
class MixedTd3Agent(tf_agent.TFAgent):
  """A TD3 Agent."""

  def __init__(self,
               time_step_spec,
               action_spec,
               actor_network,
               q_network,
               actor_optimizer,
               critic_optimizer,
               exploration_noise_std=0.1,
               boltzmann_temperature=None,
               epsilon_greedy=0.1,
               q_network_2=None,
               target_actor_network=None,
               target_q_network=None,
               target_q_network_2=None,
               target_update_tau=1.0,
               target_update_period=1,
               actor_update_period=1,
               dqda_clipping=None,
               td_errors_loss_fn=None,
               gamma=1.0,
               reward_scale_factor=1.0,
               target_policy_noise=0.2,
               target_policy_noise_clip=0.5,
               gradient_clipping=None,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               train_step_counter=None,
               action_params_mask=None,
               n_step_update=1,
               name=None):
    """Creates a Td3Agent Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A namedtuple of nested BoundedTensorSpec representing the actions.
      actor_network: A tf_agents.network.Network to be used by the agent. The
        network will be called with call(observation, step_type).
      q_network: A tf_agents.network.Network to be used by the agent. The
        network will be called with call(observation, action, step_type).
      actor_optimizer: The default optimizer to use for the actor network.
      critic_optimizer: The default optimizer to use for the critic network.
      exploration_noise_std: Scale factor on exploration policy noise.
      q_network_2: (Optional.)  A `tf_agents.network.Network` to be used as
        the second critic network during Q learning.  The weights from
        `q_network` are copied if this is not provided.
      target_actor_network: (Optional.)  A `tf_agents.network.Network` to be
        used as the target actor network during Q learning. Every
        `target_update_period` train steps, the weights from `actor_network` are
        copied (possibly withsmoothing via `target_update_tau`) to `
        target_actor_network`.  If `target_actor_network` is not provided, it is
        created by making a copy of `actor_network`, which initializes a new
        network with the same structure and its own layers and weights.
        Performing a `Network.copy` does not work when the network instance
        already has trainable parameters (e.g., has already been built, or when
        the network is sharing layers with another).  In these cases, it is up
        to you to build a copy having weights that are not shared with the
        original `actor_network`, so that this can be used as a target network.
        If you provide a `target_actor_network` that shares any weights with
        `actor_network`, a warning will be logged but no exception is thrown.
      target_q_network: (Optional.) Similar network as target_actor_network
        but for the q_network. See documentation for target_actor_network.
      target_q_network_2: (Optional.) Similar network as
        target_actor_network but for the q_network_2. See documentation for
        target_actor_network. Will only be used if 'q_network_2' is also
        specified.
      target_update_tau: Factor for soft update of the target networks.
      target_update_period: Period for soft update of the target networks.
      actor_update_period: Period for the optimization step on actor network.
      dqda_clipping: A scalar or float clips the gradient dqda element-wise
        between [-dqda_clipping, dqda_clipping]. Default is None representing no
        clippiing.
      td_errors_loss_fn:  A function for computing the TD errors loss. If None,
        a default value of elementwise huber_loss is used.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      target_policy_noise: Scale factor on target action noise
      target_policy_noise_clip: Value to clip noise.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      action_params_mask: A mask of continuous parameter actions for discrete action
      name: The name of this agent. All variables in this module will fall
        under that name. Defaults to the class name.
    """
    tf.Module.__init__(self, name=name)
    self._actor_network = actor_network
    self._target_actor_network = common.maybe_copy_target_network_with_checks(
        self._actor_network, target_actor_network, 'TargetActorNetwork')

    # critic network here is Q-network
    self._q_network_1 = q_network
    self._target_q_network_1 = (
        common.maybe_copy_target_network_with_checks(self._q_network_1,
                                                     target_q_network,
                                                     'TargetCriticNetwork1'))

    if q_network_2 is not None:
      self._q_network_2 = q_network_2
    else:
      self._q_network_2 = q_network.copy(name='CriticNetwork2')
      # Do not use target_q_network_2 if q_network_2 is None.
      target_q_network_2 = None
    self._target_q_network_2 = (
        common.maybe_copy_target_network_with_checks(self._q_network_2,
                                                     target_q_network_2,
                                                     'TargetCriticNetwork2'))
    self._actor_optimizer = actor_optimizer
    self._critic_optimizer = critic_optimizer

    self._exploration_noise_std = exploration_noise_std
    self._epsilon_greedy = epsilon_greedy
    self._boltzmann_temperature = boltzmann_temperature
    self._target_update_tau = target_update_tau
    self._target_update_period = target_update_period
    self._actor_update_period = actor_update_period
    self._dqda_clipping = dqda_clipping
    self._td_errors_loss_fn = (
        td_errors_loss_fn or common.element_wise_huber_loss)
    self._gamma = gamma
    self._reward_scale_factor = reward_scale_factor
    self._target_policy_noise = target_policy_noise
    self._target_policy_noise_clip = target_policy_noise_clip
    self._gradient_clipping = gradient_clipping

    self._update_target = self._get_target_updater(
        target_update_tau, target_update_period)

    policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec.actor_network,
        actor_network=self._actor_network, clip=True)
    policy = mixed_q_policy.MixedQPolicy(policy, time_step_spec=time_step_spec,
        action_spec=action_spec.q_network, q_network=q_network)

    collect_policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec.actor_network,
        actor_network=self._actor_network, clip=False)
    collect_policy = gaussian_policy.GaussianPolicy(
        collect_policy,
        scale=self._exploration_noise_std,
        clip=True)
    collect_policy = mixed_q_policy.MixedQPolicy(collect_policy, time_step_spec=time_step_spec,
        action_spec=action_spec.q_network, q_network=q_network)
    if boltzmann_temperature is not None:
      collect_policy = boltzmann_policy.BoltzmannPolicy(
          collect_policy, temperature=self._boltzmann_temperature)
    else:
      collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
          collect_policy, epsilon=self._epsilon_greedy)

    # Create self._target_greedy_policy in order to compute target Q-values.

    target_policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec.actor_network,
        actor_network=self._target_actor_network, clip=True)
    target_policy = mixed_q_policy.MixedQPolicy(target_policy, time_step_spec=time_step_spec,
        action_spec=action_spec.q_network, q_network=self._target_q_network_1)
    self._target_greedy_policy = greedy_policy.GreedyPolicy(target_policy)
    self._action_params_mask = action_params_mask
    self._n_step_update = n_step_update
    if action_spec.actor_network is not None and action_params_mask is None:
        raise ValueError("action_params_mask is required for actor network")

    super(MixedTd3Agent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=2 if not self._actor_network.state_spec else None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

  def _initialize(self):
    """Initialize the agent.

    Copies weights from the actor and critic networks to the respective
    target actor and critic networks.
    """
    common.soft_variables_update(
        self._q_network_1.variables,
        self._target_q_network_1.variables,
        tau=1.0)
    common.soft_variables_update(
        self._q_network_2.variables,
        self._target_q_network_2.variables,
        tau=1.0)
    common.soft_variables_update(
        self._actor_network.variables,
        self._target_actor_network.variables,
        tau=1.0)

    ##TODO: override _check_trajectory_dimensions
  def _check_trajectory_dimensions(self, experience):
    """Checks the given Trajectory for batch and time outer dimensions."""
    if not nest_utils.is_batched_nested_tensors(
        experience, self.collect_data_spec,
        num_outer_dims=self._num_outer_dims):
      debug_str_1 = tf.nest.map_structure(lambda tp: tp.shape, experience)
      debug_str_2 = tf.nest.map_structure(lambda spec: spec.shape,
                                          self.collect_data_spec)

      if self._num_outer_dims == 2:
        raise ValueError(
            "All of the Tensors in `experience` must have two outer "
            "dimensions: batch size and time. Specifically, tensors should be "
            "shaped as [B x T x ...].\n"
            "Full shapes of experience tensors:\n{}.\n"
            "Full expected shapes (minus outer dimensions):\n{}.".format(
                debug_str_1, debug_str_2))
      else:
        # self._num_outer_dims must be 1.
        raise ValueError(
            "All of the Tensors in `experience` must have a single outer "
            "batch_size dimension. If you also want to include an outer time "
            "dimension, set num_outer_dims=2 when initializing your agent.\n"
            "Full shapes of experience tensors:\n{}.\n"
            "Full expected shapes (minus batch_size dimension):\n{}.".format(
                debug_str_1, debug_str_2))

    # If we have a time dimension and a train_sequence_length, make sure they
    # match.
    if self._num_outer_dims == 2 and self.train_sequence_length is not None:
      def check_shape(t):  # pylint: disable=invalid-name
        if t.shape[1] != self.train_sequence_length:
          debug_str = tf.nest.map_structure(lambda tp: tp.shape, experience)
          raise ValueError(
              "One of the Tensors in `experience` has a time axis dim value "
              "'%s', but we require dim value '%d'. Full shape structure of "
              "experience:\n%s" %
              (t.shape[1], self.train_sequence_length, debug_str))

      tf.nest.map_structure(check_shape, experience)
      
  def _get_target_updater(self, tau=1.0, period=1):
    """Performs a soft update of the target network parameters.

    For each weight w_s in the original network, and its corresponding
    weight w_t in the target network, a soft update is:
    w_t = (1- tau) x w_t + tau x w_s

    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
      period: Step interval at which the target networks are updated.
    Returns:
      A callable that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_targets'):
      def update():  # pylint: disable=missing-docstring
        # TODO(b/124381161): What about observation normalizer variables?
        critic_update_1 = common.soft_variables_update(
            self._q_network_1.variables,
            self._target_q_network_1.variables, tau)
        critic_update_2 = common.soft_variables_update(
            self._q_network_2.variables,
            self._target_q_network_2.variables, tau)
        actor_update = common.soft_variables_update(
            self._actor_network.variables, self._target_actor_network.variables,
            tau)
        return tf.group(critic_update_1, critic_update_2, actor_update)

      return common.Periodically(update, period, 'update_targets')

  def _experience_to_transitions(self, experience):
    transitions = trajectory.to_transition(experience)

    # Remove time dim if we are not using a recurrent network.
    if not self._actor_network.state_spec:
      transitions = tf.nest.map_structure(lambda x: tf.squeeze(x, [1]),
                                          transitions)

    time_steps, policy_steps, next_time_steps = transitions
    actions = policy_steps.action
    return time_steps, actions, next_time_steps

  def _train(self, experience, weights=None):
    # TODO(b/120034503): Move the conversion to transitions to the base class.

    if self._n_step_update == 1:
        time_steps, actions, next_time_steps = self._experience_to_transitions(
            experience)
    else:
        # To compute n-step returns, we need the first time steps, the first
        # actions, and the last time steps. Therefore we extract the first and
        # last transitions from our Trajectory.
        first_two_steps = tf.nest.map_structure(lambda x: x[:, :2], experience)
        last_two_steps = tf.nest.map_structure(lambda x: x[:, -2:], experience)
        time_steps, actions, _ = self._experience_to_transitions(first_two_steps)
        _, _, next_time_steps = self._experience_to_transitions(last_two_steps)

    trainable_critic_variables = (
        self._q_network_1.trainable_variables +
        self._q_network_2.trainable_variables)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_critic_variables, ('No trainable critic variables to '
                                          'optimize.')
      tape.watch(trainable_critic_variables)
      critic_loss = self.critic_loss(experience, weights=weights)
    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
    self._apply_gradients(critic_grads, trainable_critic_variables,
                          self._critic_optimizer)

    trainable_actor_variables = self._actor_network.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_actor_variables, ('No trainable actor variables to '
                                         'optimize.')
      tape.watch(trainable_actor_variables)
      discrete_actions = actions
      actor_loss = self.actor_loss(time_steps, discrete_actions, weights=weights)
    tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')

    # We only optimize the actor every actor_update_period training steps.
    def optimize_actor():
      actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
      return self._apply_gradients(actor_grads, trainable_actor_variables,
                                   self._actor_optimizer)

    remainder = tf.math.mod(self.train_step_counter, self._actor_update_period)
    tf.cond(
        pred=tf.equal(remainder, 0), true_fn=optimize_actor, false_fn=tf.no_op)

    self.train_step_counter.assign_add(1)
    self._update_target()

    # TODO(b/124382360): Compute per element TD loss and return in loss_info.
    total_loss = actor_loss + critic_loss

    return tf_agent.LossInfo(total_loss,
                             Td3Info(actor_loss, critic_loss))

  def _apply_gradients(self, gradients, variables, optimizer):
    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = tuple(zip(gradients, variables))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(
          grads_and_vars, self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    return optimizer.apply_gradients(grads_and_vars)

  @common.function
  def critic_loss(self, experience, gamma=1.0, weights=None):
    """Computes the critic loss for TD3 training.

    Args:
      experience: A batch of timesteps.
      gamma: reward discount factor
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.

    Returns:
      critic_loss: A scalar critic loss.
    """
    with tf.name_scope('critic_loss'):

      self._check_trajectory_dimensions(experience)

      if self._n_step_update == 1:
          time_steps, actions, next_time_steps = self._experience_to_transitions(
              experience)
      else:
          # To compute n-step returns, we need the first time steps, the first
          # actions, and the last time steps. Therefore we extract the first and
          # last transitions from our Trajectory.
          first_two_steps = tf.nest.map_structure(lambda x: x[:, :2], experience)
          last_two_steps = tf.nest.map_structure(lambda x: x[:, -2:], experience)
          time_steps, actions, _ = self._experience_to_transitions(first_two_steps)
          _, _, next_time_steps = self._experience_to_transitions(last_two_steps)

      # Target q-values are the min of the two networks
      target_q_values_1 = self._compute_next_q_values(self._target_q_network_1, next_time_steps)
      target_q_values_2 = self._compute_next_q_values(self._target_q_network_2, next_time_steps)

      target_q_values = tf.minimum(target_q_values_1, target_q_values_2)

      if self._n_step_update == 1:
        # Special case for n = 1 to avoid a loss of performance.
        td_targets = compute_td_targets(
            target_q_values,
            rewards=self._reward_scale_factor * next_time_steps.reward,
            discounts=gamma * next_time_steps.discount)
      else:
        # When computing discounted return, we need to throw out the last time
        # index of both reward and discount, which are filled with dummy values
        # to match the dimensions of the observation.
        rewards = self._reward_scale_factor * experience.reward[:, :-1]
        discounts = gamma * experience.discount[:, :-1]

        td_targets = value_ops.discounted_return(
            rewards=rewards,
            discounts=discounts,
            final_value=target_q_values,
            time_major=False,
            provide_all_returns=False)

       # td_targets = tf.stop_gradient(
       #    self._reward_scale_factor * next_time_steps.reward +
       #    self._gamma * next_time_steps.discount * target_q_values)

      pred_td_targets_1, _ = self._compute_q_values(self._q_network_1, time_steps, actions)
      pred_td_targets_2, _ = self._compute_q_values(self._q_network_2, time_steps, actions)
      pred_td_targets_all = [pred_td_targets_1, pred_td_targets_2]

      if self._debug_summaries:
        tf.compat.v2.summary.histogram(
            name='td_targets', data=td_targets, step=self.train_step_counter)
        with tf.name_scope('td_targets'):
          tf.compat.v2.summary.scalar(
              name='mean',
              data=tf.reduce_mean(input_tensor=td_targets),
              step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              name='max',
              data=tf.reduce_max(input_tensor=td_targets),
              step=self.train_step_counter)
          tf.compat.v2.summary.scalar(
              name='min',
              data=tf.reduce_min(input_tensor=td_targets),
              step=self.train_step_counter)

        for td_target_idx in range(2):
          pred_td_targets = pred_td_targets_all[td_target_idx]
          td_errors = td_targets - pred_td_targets
          with tf.name_scope('critic_net_%d' % (td_target_idx + 1)):
            tf.compat.v2.summary.histogram(
                name='td_errors', data=td_errors, step=self.train_step_counter)
            tf.compat.v2.summary.histogram(
                name='pred_td_targets',
                data=pred_td_targets,
                step=self.train_step_counter)
            with tf.name_scope('td_errors'):
              tf.compat.v2.summary.scalar(
                  name='mean',
                  data=tf.reduce_mean(input_tensor=td_errors),
                  step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                  name='mean_abs',
                  data=tf.reduce_mean(input_tensor=tf.abs(td_errors)),
                  step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                  name='max',
                  data=tf.reduce_max(input_tensor=td_errors),
                  step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                  name='min',
                  data=tf.reduce_min(input_tensor=td_errors),
                  step=self.train_step_counter)
            with tf.name_scope('pred_td_targets'):
              tf.compat.v2.summary.scalar(
                  name='mean',
                  data=tf.reduce_mean(input_tensor=pred_td_targets),
                  step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                  name='max',
                  data=tf.reduce_max(input_tensor=pred_td_targets),
                  step=self.train_step_counter)
              tf.compat.v2.summary.scalar(
                  name='min',
                  data=tf.reduce_min(input_tensor=pred_td_targets),
                  step=self.train_step_counter)

      critic_loss = (self._td_errors_loss_fn(td_targets, pred_td_targets_1)
                     + self._td_errors_loss_fn(td_targets, pred_td_targets_2))
      if nest_utils.is_batched_nested_tensors(
          time_steps, self.time_step_spec, num_outer_dims=2):
        # Sum over the time dimension.
        critic_loss = tf.reduce_sum(input_tensor=critic_loss, axis=1)

      if weights is not None:
        critic_loss *= weights

      return tf.reduce_mean(input_tensor=critic_loss)

  def _compute_q_values(self, q_network, time_steps, discrete_actions):

    continuous_action_values, _ = self._actor_network(time_steps.observation,
                                                      time_steps.step_type)
    # noisy_target_action_values = tf.nest.map_structure(self._add_noise_to_action,
    #                                              target_action_values)

    time_step_obs = tf.nest.flatten(time_steps.observation) + [continuous_action_values]
    if isinstance(q_network, QNetwork):
        time_step_obs = tf.concat(time_step_obs, axis=-1)
    q_values, _ = q_network(time_step_obs,
                                  time_steps.step_type)

    # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
    # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
    multi_dim_actions = tf.nest.flatten(self._action_spec.q_network)[0].shape.ndims > 0
    return common.index_with_actions(
        q_values,
        tf.cast(discrete_actions, dtype=tf.int32),
        multi_dim_actions=multi_dim_actions), continuous_action_values

  def _compute_next_q_values(self, target_q_network, next_time_steps):
    """Compute the q value of the next state for TD error computation.

    Args:
      next_time_steps: A batch of next timesteps

    Returns:
      A tensor of Q values for the given next state.
    """

    next_target_continuous_action_values, _ = self._target_actor_network(
        next_time_steps.observation, next_time_steps.step_type)
    noisy_target_action_values = tf.nest.map_structure(self._add_noise_to_action,
                                                 next_target_continuous_action_values)

    time_step_obs = tf.nest.flatten(next_time_steps.observation) + [noisy_target_action_values]
    if isinstance(target_q_network, QNetwork):
        time_step_obs = tf.concat(time_step_obs, axis=-1)
    next_target_q_values, _ = target_q_network(
        time_step_obs, next_time_steps.step_type)
    batch_size = (
        next_target_q_values.shape[0] or tf.shape(next_target_q_values)[0])
    dummy_state = self._target_greedy_policy.get_initial_state(batch_size)
    # Find the greedy actions using our target greedy policy. This ensures that
    # masked actions are respected and helps centralize the greedy logic.
    greedy_discrete_actions = self._target_greedy_policy.action(
        next_time_steps, dummy_state).action

    # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
    # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
    multi_dim_actions = tf.nest.flatten(self._action_spec.q_network)[0].shape.ndims > 0
    return common.index_with_actions(
        next_target_q_values,
        greedy_discrete_actions,
        multi_dim_actions=multi_dim_actions)

  # Add gaussian noise to each action before computing target q values
  def _add_noise_to_action(self, action):  # pylint: disable=missing-docstring
      dist = tfp.distributions.Normal(loc=tf.zeros_like(action),
                                      scale=self._target_policy_noise * \
                                            tf.ones_like(action))
      noise = dist.sample()
      noise = tf.clip_by_value(noise, -self._target_policy_noise_clip,
                               self._target_policy_noise_clip)
      return action + noise

  @common.function
  def actor_loss(self, time_steps, discrete_actions, weights=None):
    """Computes the actor_loss for TD3 training.

    Args:
      time_steps: A batch of timesteps.
      discrete_actions: A tensor of discrete action arguments.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.
      # TODO(b/124383618): Add an action norm regularizer.
    Returns:
      actor_loss: A scalar actor loss.
    """
    with tf.name_scope('actor_loss'):
      with tf.GradientTape(watch_accessed_variables=True) as tape:
        q_values, continuous_actions = self._compute_q_values(self._q_network_1, time_steps, discrete_actions)
        continuous_actions = tf.nest.flatten(continuous_actions)
        tape.watch(continuous_actions)

      dqdas = tape.gradient([q_values], continuous_actions)
      actor_losses = []
      for dqda, cont_action in zip(dqdas, continuous_actions):
        if self._dqda_clipping is not None:
          dqda = tf.clip_by_value(dqda, -1 * self._dqda_clipping,
                                  self._dqda_clipping)
        # mask unrelevant continuous actions for each discrete action
        multi_dim_actions = tf.nest.flatten(self._action_spec.q_network)[0].shape.ndims > 0
        if multi_dim_actions:
            raise NotImplementedError("multidimensional action space is not supported")
        discrete_actions_shape = tf.shape(discrete_actions)
        cont_action_mask = tf.cast(tf.gather_nd(self._action_params_mask, tf.reshape(discrete_actions, [-1, 1])),
                                   tf.float32)
        cont_action_mask = tf.reshape(cont_action_mask,
                                      tf.concat([discrete_actions_shape, [-1]], axis=-1))
        loss = common.element_wise_squared_loss(
            tf.stop_gradient(dqda + cont_action), cont_action)

        tf.nest.assert_same_structure(loss, cont_action_mask)
        loss = loss * cont_action_mask
        if nest_utils.is_batched_nested_tensors(
            time_steps, self.time_step_spec, num_outer_dims=2):
          # Sum over the time dimension.
          loss = tf.reduce_sum(loss, axis=1)
        if weights is not None:
          loss *= weights
        loss = tf.reduce_mean(loss)
        actor_losses.append(loss)

      actor_loss = tf.add_n(actor_losses)

      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
            name='actor_loss', data=actor_loss, step=self.train_step_counter)

    return actor_loss
