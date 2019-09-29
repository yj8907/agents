# !/usr/bin/env python3
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
from tf_agents.policies import sequential_policy

from tf_agents.networks.q_network import QNetwork

from tf_agents.agents import tf_agent

from tf_agents.policies import hetero_q_policy
from tf_agents.policies import gaussian_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import value_ops

import numpy as np
NEG_INF = tf.constant(-np.inf)

import copy

class Td3Info(collections.namedtuple(
    'Td3Info', ('critic_loss',))):
    pass


def compute_td_targets(next_q_values, rewards, discounts):
    return tf.stop_gradient(rewards + discounts * next_q_values)


@gin.configurable
class Td3DqnAgent(tf_agent.TFAgent):
    """A TD3 Agent."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 q_networks,
                 critic_optimizer,
                 exploration_noise_std=0.1,
                 boltzmann_temperature=None,
                 epsilon_greedy=0.1,
                 q_networks_2=None,
                 target_q_networks=None,
                 target_q_networks_2=None,
                 target_update_tau=1.0,
                 target_update_period=1,
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
          action_spec: A list of tf_agents.network.Network to be used by the agent. The
            network will be called with call(observation, action, step_type).
          q_networks: A tf_agents.network.Network to be used by the agent. The
            network will be called with call(observation, action, step_type).
          critic_optimizer: The default optimizer to use for the critic network.
          exploration_noise_std: Scale factor on exploration policy noise.
          q_networks_2: (Optional.)  A `tf_agents.network.Network` to be used as
            the second critic network during Q learning.  The weights from
            `q_network` are copied if this is not provided.
          target_q_networks: (Optional.)  A `tf_agents.network.Network` to be
            used as the target Q network during Q learning. Every
            `target_update_period` train steps, the weights from `q_networks` are
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
          target_q_networks_2: (Optional.) Similar network as target_actor_network
            but for the q_network. See documentation for target_actor_network.
          target_update_tau: Factor for soft update of the target networks.
          target_update_period: Period for soft update of the target networks.
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

        # critic network here is Q-network
        if not isinstance(q_networks, (list, tuple)):
            q_networks = (q_networks,)
        self._q_network_1 = q_networks

        if target_q_networks is not None and not isinstance(target_q_networks, (list, tuple)):
            target_q_networks = (target_q_networks,)
        if target_q_networks is None:
            target_q_networks = [None]*len(self._q_network_1)
        assert len(self._q_network_1) == len(target_q_networks)

        self._target_q_network_1 = [common.maybe_copy_target_network_with_checks(
                    q, target_q, 'Target'+q.name+'_1')
                for q, target_q in zip(self._q_network_1, target_q_networks)]

        if q_networks_2 is not None:
            self._q_network_2 = q_networks_2
        else:
            self._q_network_2 = [q.copy(name=q.name+'_2') for q in q_networks]
            # Do not use target_q_network_2 if q_network_2 is None.
            target_q_networks_2 = None

        if target_q_networks_2 is not None and not isinstance(target_q_networks, (list, tuple)):
            target_q_networks_2 = (target_q_networks_2,)
        if target_q_networks_2 is None:
            target_q_networks_2 = [None]*len(self._q_network_2)

        self._target_q_network_2 = [ common.maybe_copy_target_network_with_checks(
                    q, target_q, 'Target'+q.name+'_2')
                for q, target_q in zip(self._q_network_2, target_q_networks_2)]

        self._critic_optimizer = critic_optimizer

        self._exploration_noise_std = exploration_noise_std
        self._epsilon_greedy = epsilon_greedy
        self._boltzmann_temperature = boltzmann_temperature
        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
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

        num_networks = len(self._q_network_1)
        if len(action_spec) != num_networks or len(time_step_spec) != num_networks:
            raise ValueError("number of networks should match number of action_spec and time_step_spec")

        policies = []
        self._q_value_policies_1, self._q_value_policies_2 = [], []
        for i, q_network in enumerate(self._q_network_1):
            use_previous_action = i != 0
            policies.append(hetero_q_policy.HeteroQPolicy(time_step_spec=time_step_spec[i], action_spec=action_spec[i],
                              mixed_q_network=q_networks[i], func_arg_mask=action_params_mask[i],
                              use_previous_action=use_previous_action, name="HeteroQPolicy_1_"+str(i)))
            self._q_value_policies_2.append(
                        hetero_q_policy.HeteroQPolicy(time_step_spec=time_step_spec[i], action_spec=action_spec[i],
                              mixed_q_network=self._q_network_2[i], func_arg_mask=action_params_mask[i],
                              use_previous_action=use_previous_action, name="HeteroQPolicy_2_"+str(i)))
        self._q_value_policies_1 = policies

        if boltzmann_temperature is not None:
            collect_policies = [boltzmann_policy.BoltzmannPolicy(
                p, temperature=self._boltzmann_temperature) for p in policies]
        else:
            raise ValueError("boltzmann_temperature should not be None")
            # collect_policies = [epsilon_greedy_policy.EpsilonGreedyPolicy(
            #     p, epsilon=self._epsilon_greedy) for p in policies]
        collect_policy = sequential_policy.SequentialPolicy(collect_policies)

        policies = [greedy_policy.GreedyPolicy(p) for p in policies]
        policy = sequential_policy.SequentialPolicy(policies)

        # Create self._target_greedy_policy in order to compute target Q-values in _compute_next_q_values.
        target_policies = []
        self._target_q_value_policies_1, self._target_q_value_policies_2 = [], []
        for i, q_network in enumerate(self._q_network_1):
            use_previous_action = i != 0
            target_policies.append(hetero_q_policy.HeteroQPolicy(
                              time_step_spec=time_step_spec[i], action_spec=action_spec[i],
                              mixed_q_network=self._target_q_network_1[i], func_arg_mask=action_params_mask[i],
                              use_previous_action=use_previous_action, name="TargetHeteroQPolicy_1"+str(i)))
            self._target_q_value_policies_2.append(hetero_q_policy.HeteroQPolicy(
                              time_step_spec=time_step_spec[i], action_spec=action_spec[i],
                              mixed_q_network=self._target_q_network_2[i], func_arg_mask=action_params_mask[i],
                              use_previous_action=use_previous_action, name="TargetHeteroQPolicy_2"+str(i)))
        self._target_q_value_policies_1 = target_policies

        self._target_q_value_policies_1 = [greedy_policy.GreedyPolicy(p) for p in self._target_q_value_policies_1]
        self._target_q_value_policies_2 = [greedy_policy.GreedyPolicy(p) for p in self._target_q_value_policies_2]

        target_policies = self._target_q_value_policies_1
        target_policy = sequential_policy.SequentialPolicy(target_policies)

        self._target_greedy_policies = target_policy

        self._action_params_mask = action_params_mask
        self._n_step_update = n_step_update

        super(Td3DqnAgent, self).__init__(
            policy.time_step_spec,
            policy.action_spec,
            policy,
            collect_policy,
            train_sequence_length=2 if not self._q_network_1[0].state_spec else None,
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
                return tf.group(critic_update_1, critic_update_2)

            return common.Periodically(update, period, 'update_targets')

    def _experience_to_transitions(self, experience):
        transitions = trajectory.to_transition(experience)

        # Remove time dim if we are not using a recurrent network.
        if not self._q_network_1[0].state_spec:
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

        # remainder = tf.math.mod(self.train_step_counter, self._actor_update_period)

        self.train_step_counter.assign_add(1)
        self._update_target()

        # TODO(b/124382360): Compute per element TD loss and return in loss_info.
        total_loss = critic_loss

        return tf_agent.LossInfo(total_loss,
                                 Td3Info(critic_loss))

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

    # @common.function
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
            # print("first pass")
            target_q_values_1 = self._compute_next_q_values(self._target_q_value_policies_1, next_time_steps)
            target_q_values_2 = self._compute_next_q_values(self._target_q_value_policies_2, next_time_steps)

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
            # print("second pass")
            pred_td_targets_1 = self._compute_q_values(self._q_value_policies_1, time_steps, actions)
            pred_td_targets_2 = self._compute_q_values(self._q_value_policies_2, time_steps, actions)
            pred_td_targets_all = [pred_td_targets_1, pred_td_targets_2]
            # print("third pass")
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
            # print("forth pass")
            return tf.reduce_mean(input_tensor=critic_loss)

    def _sequential_network_activation(self, policies, time_step_input, actions=None):

        """
        compute Q-value conditioned on previous actions

        :param policies: list of HeteroQPolicy object
        :param actions: dict of actions extracted from replay buffer, which complies with action_spec
                'raw': a Tensor of size [batch_size, num_policies],
                'transformed': a Tensor of size [batch_size, num_policies, 3]
        :return: list of Q values corresponding to each policy
        """

        time_step = tf.nest.map_structure(lambda x: x, time_step_input)

        previous_action_key = 'previous_action'
        func_action_key = 'func_action'
        available_actions_key = 'available_actions'

        raw_action_key, transformed_action_key = 'raw', 'transformed'
        assert previous_action_key not in time_step.observation

        discrete_action_key, continuous_action_key = 'discrete', 'continuous'
        num_policies = len(policies)
        q_values = []

        raw_action = []
        transformed_actions = []
        action_transform_key = 'inverse_transform_action'

        for i, policy in enumerate(policies):
            if i == 0:
                assert available_actions_key in time_step.observation
            elif available_actions_key in time_step.observation:
                time_step.observation.pop(available_actions_key)

            # print((i, policy))
            action_step, distribution_step = policy.action_distribution(time_step)

            if action_transform_key in type(policy).__dict__.keys():
                action = policy.inverse_transform_action(action_step.action)
            else:
                assert 'wrapped_policy' in type(policy).__dict__.keys()
                assert action_transform_key in type(policy.wrapped_policy).__dict__.keys()
                action = policy.wrapped_policy.inverse_transform_action(action_step.action)

            raw_action.append(action_step.action)
            transformed_actions.append(tf.expand_dims(action, axis=1))

            current_actions = dict()
            if actions is None:
                current_actions[transformed_action_key] = action
            else:
                current_actions[raw_action_key] = actions[raw_action_key][:, i]
                current_actions[transformed_action_key] = actions[transformed_action_key][:, i, :]

            # create function action tensor
            if i == 0:
                time_step.observation[func_action_key] = dict()
                time_step.observation[func_action_key][discrete_action_key] = \
                    tf.cast(current_actions[transformed_action_key][:, 0], tf.int32)

            # create previous action tensor
            time_step.observation[previous_action_key] = dict()
            if i < num_policies - 1 and \
                    discrete_action_key in policies[i+1].time_step_spec.observation[previous_action_key]:
                time_step.observation[previous_action_key][discrete_action_key] = \
                        tf.cast(current_actions[transformed_action_key][:, 0], tf.int32)
            if i < num_policies - 1 and \
                    continuous_action_key in policies[i+1].time_step_spec.observation[previous_action_key]:
                time_step.observation[previous_action_key][continuous_action_key] = \
                        current_actions[transformed_action_key][:, 1:]

            q_values.append(distribution_step.action.logits)
        time_step.observation.pop(previous_action_key)
        time_step.observation.pop(func_action_key)

        raw_action = tf.concat(raw_action, axis=-1)
        transformed_actions = tf.concat(transformed_actions, axis=1)

        if actions is None:
            actions = dict()
            actions[raw_action_key] = raw_action
            actions[transformed_action_key] = transformed_actions

        return q_values, actions

    def _compute_q_values(self, policies, time_steps, actions):
        """Compute the q value of the current state/action for TD error computation.

        Args:
          policies: list of HeteroQPolicy object
          time_steps: A batch of current timesteps
          actions: A batch of actions
        Returns:
          A tensor of Q values for the given next state.
        """

        q_values_seq, _ = self._sequential_network_activation(policies, time_steps, actions)

        action_key = 'raw'
        raw_actions = tf.unstack(actions[action_key], axis=1)
        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        multi_dim_actions = False
        value = [common.index_with_actions(
            self._append2logits(tf.squeeze(q_values, axis=1)),
            tf.cast(act, dtype=tf.int32),
            multi_dim_actions=multi_dim_actions) for q_values, act in zip(q_values_seq, raw_actions)]
        # due to dist.mode() in GreedyPolicy, 0 is selected for masked action. So we need to remove -inf
        value = [tf.where(tf.equal(v, NEG_INF), 0.0, v) for v in value]
        value = tf.add_n(value)

        return value

    def _compute_next_q_values(self, target_policies, time_steps):
        """Compute the q value of the next state for TD error computation.

        Args:
          policies: list of target HeteroQPolicy object
          time_steps: A batch of current timesteps
        Returns:
          A tensor of Q values for the given next state.
        """

        q_values_seq, actions = self._sequential_network_activation(target_policies, time_steps)

        action_key = 'raw'
        raw_actions = tf.unstack(actions[action_key], axis=1)
        multi_dim_actions = False
        value = [common.index_with_actions(
            self._append2logits(tf.squeeze(q_values, axis=1)),
            tf.cast(act, dtype=tf.int32),
            multi_dim_actions=multi_dim_actions) for q_values, act in zip(q_values_seq, raw_actions)]
        # due to dist.mode() in GreedyPolicy, 0 is selected for masked action. So we need to remove -inf
        value = [tf.where(tf.equal(v, NEG_INF), 0.0, v) for v in value]
        value = tf.add_n(value)

        return value

    def _append2logits(self, logits):
        """
        Add zeros to last position of logits to alllow indexing of masked actions
        :param logits: a Tensor of shape [batch_size, num_dims]
        :return: a Tensor of shape [batch_size, num_dims] with zeros appened to last dimension
        """

        assert logits.shape.ndims == 2
        zeros = tf.zeros((logits.shape[0], 1), tf.float32)
        return tf.concat([logits, zeros], axis=-1)

    # Add gaussian noise to each action before computing target q values
    def _add_noise_to_action(self, action):  # pylint: disable=missing-docstring
        dist = tfp.distributions.Normal(loc=tf.zeros_like(action),
                                        scale=self._target_policy_noise * \
                                              tf.ones_like(action))
        noise = dist.sample()
        noise = tf.clip_by_value(noise, -self._target_policy_noise_clip,
                                 self._target_policy_noise_clip)
        return action + noise


