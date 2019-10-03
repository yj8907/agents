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
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.distributions import shifted_categorical
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
from tf_agents.networks.q_network import QNetwork

from tf_agents.utils import common

from tensorflow.python.framework import tensor_spec
import copy

tfd = tfp.distributions

@gin.configurable
class HeteroQPolicy(tf_policy.Base):
    """Class to build Q-Policies."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 mixed_q_network,
                 func_arg_mask=None,
                 emit_log_probability=False,
                 spatial_names=("screen", "minimap"),
                 structured_names=("structured",),
                 use_previous_action=False,
                 name=None):
        """Builds a Q-Policy given a q_network.

        Args:
          time_step_spec: A `TimeStep` spec of the expected time_steps.
          action_spec: A nest of BoundedTensorSpec representing the actions.
          mixed_q_network: An instance of a `tf_agents.network.Network`,
            callable via `network(observation, step_type) -> (output, final_state)`.
          emit_log_probability: Whether to emit log-probs in info of `PolicyStep`.
          func_arg_mask: A mask Tensor to allow action specific selection of arguments.
          name: The name of this policy. All variables in this module will fall
            under that name. Defaults to the class name.

        Raises:
          ValueError: If `q_network.action_spec` exists and is not compatible with
            `action_spec`.
          NotImplementedError: If `action_spec` contains more than one
            `BoundedTensorSpec`.
        """
        network_action_spec = getattr(mixed_q_network, 'action_spec', None)

        self._available_actions_key = 'available_actions'
        self._func_action_key = 'func_action'
        self._previous_action_key = "previous_action"

        if use_previous_action != (self._previous_action_key in time_step_spec.observation):
            raise ValueError("use_previous_action is not consistent with time_step_spec.observation previous_action")
        if use_previous_action != (self._func_action_key in time_step_spec.observation):
            raise ValueError("use_previous_action is not consistent with time_step_spec.observation func_action")
        if use_previous_action != (func_arg_mask is not None):
            raise ValueError("use_previous_action is not consistent with func_arg_mask")

        if network_action_spec is not None:
            if not isinstance(network_action_spec, (list, dict, tuple)):
                if not action_spec.is_compatible_with(network_action_spec):
                    raise ValueError(
                        'action_spec must be compatible with mixed_q_network.action_spec; '
                        'instead got action_spec=%s, mixed_q_network.action_spec=%s' % (
                            action_spec, network_action_spec))
            else:
                compatible = [a.is_compatible_with(b)
                      for a, b in zip(tf.nest.flatten(action_spec), tf.nest.flatten(network_action_spec))]
                if not all(compatible):
                    raise ValueError(
                        'action_spec must be compatible with mixed_q_network.action_spec; ')

        if func_arg_mask is not None:
            # create zero mask for previous masked action
            assert len(func_arg_mask.shape) == 2
            self._func_arg_mask = copy.deepcopy(func_arg_mask)
            self._func_arg_mask = np.concatenate([self._func_arg_mask,
                                                  np.zeros((1, self._func_arg_mask.shape[1]), np.float32)],
                                                 axis=0)
        else:
            self._func_arg_mask = copy.deepcopy(func_arg_mask)

        self._mixed_q_network = mixed_q_network
        self._spatial_names = spatial_names
        self._structured_names = structured_names
        self._use_previous_action = use_previous_action
        self._discrete_action_key, self._continuous_action_key = 'discrete', 'continuous'

        self._keyed_action_spec = action_spec
        self._total_num_actions = None
        action_spec = self._transform_action_spec(time_step_spec, action_spec,
                                                  spatial_names, structured_names)

        super(HeteroQPolicy, self).__init__(
            time_step_spec,
            action_spec,
            policy_state_spec=mixed_q_network.state_spec,
            clip=False,
            emit_log_probability=emit_log_probability,
            name=name)

        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            print('hetero Q policy supports action BoundedTensorSpec with size > 1')
        # We need to maintain the flat action spec for dtype, shape and range.
        self._flat_action_spec = flat_action_spec

        self._action_lookup = []
        self._build_action_lookup()

    @property
    def keyed_action_spec(self):
        return self._keyed_action_spec

    def _variables(self):
        return self._q_network.variables

    @property
    def total_num_actions(self):
        return self._total_num_actions

    def _transform_action_spec(self, time_step_spec, action_spec, spatial_names, structured_names):

        num_actions = 0
        for name in spatial_names:
            assert name in time_step_spec.observation
            height = time_step_spec.observation[name][0].shape[0]
            width = time_step_spec.observation[name][0].shape[1]
            if name in action_spec:
                num_actions += height*width*(action_spec[name].maximum[0]-action_spec[name].minimum[0]+1)
        for name in structured_names:
            if name in action_spec:
                num_actions += action_spec[name].maximum[0]-action_spec[name].minimum[0]+1

        # include no-op action
        self._total_num_actions = num_actions + 1

        return tensor_spec.BoundedTensorSpec(
            shape=(), dtype=np.int32, name='combined_action_spec',
            minimum=(0,),
            maximum=(num_actions - 1,))

    def _build_action_lookup(self):

        num_actions = 0
        num_action_types = 0
        for name in self._spatial_names:
            assert name in self._time_step_spec.observation

            if name in self._keyed_action_spec:
                num_types = self._keyed_action_spec[name].maximum[0] - self._keyed_action_spec[name].minimum[0] + 1
                height = self._time_step_spec.observation[name][0].shape[0]
                width = self._time_step_spec.observation[name][0].shape[1]

                # iterate through flattened q value index
                for i in range(height):
                    for j in range(width):
                        for k in range(num_types):
                            # normalize coordinates
                            # self._action_lookup[num_actions] = (float(num_action_types),
                            #                                     (i/float(height), j/float(width)))
                            self._action_lookup.append([float(num_action_types+k), i/float(height), j/float(width)])
                            num_actions += 1
                num_action_types += num_types

        for name in self._structured_names:
            if name in self._keyed_action_spec:
                num_types = self._keyed_action_spec[name].maximum[0]-self._keyed_action_spec[name].minimum[0]+1
                for k in range(num_types):
                    # self._action_lookup[num_actions] = (float(num_action_types), (0.0, 0.0))
                    self._action_lookup.append([float(num_action_types), 0.0, 0.0])
                    num_actions += 1
                    num_action_types += 1

        # embed masked action at the end
        # self._action_lookup[num_actions] = (float(num_action_types), (0.0, 0.0))
        self._action_lookup.append([float(num_action_types), 0.0, 0.0])
        self._action_lookup = tf.convert_to_tensor(self._action_lookup)

        if self._total_num_actions - 1 != num_actions:
            raise ValueError("total number of actions don't match between action lookup build and "
                             "action spec transform")
        return

    def inverse_transform_action(self, action):

        if action.shape.ndims > 1:
            assert action.shape[1] == 1
            action = tf.squeeze(action, axis=-1)

        transformed_actions = []
        action_per_batch = tf.unstack(action)
        for act in action_per_batch:
            # tf.convert_to_tensor([act, self._total_num_actions])
            assert_op = tf.Assert(tf.less_equal(act, self._total_num_actions - 1), [act, self._total_num_actions])
            with tf.control_dependencies([assert_op]):
                # if act.numpy() > self._total_num_actions - 1:
                #     raise ValueError("action index exceeds total number of actions")
                # act = tf.convert_to_tensor(tf.nest.flatten(self._action_lookup[act.numpy()]))
                assert act.shape.ndims < 2
                act = tf.gather(self._action_lookup, act)
                act = tf.reshape(act, [1, -1])
                transformed_actions.append(act)
        transformed_actions = tf.concat(transformed_actions, axis=0)

        return transformed_actions

    def _distribution(self, time_step, policy_state):
        # In DQN, we always either take a uniformly random action, or the action
        # with the highest Q-value. However, to support more complicated policies,
        # we expose all Q-values as a categorical distribution with Q-values as
        # logits, and apply the GreedyPolicy wrapper in dqn_agent.py to select the
        # action with the highest Q-value.

        neg_inf = tf.constant(-np.inf, dtype=tf.float32)
        previous_action = time_step.observation[self._previous_action_key] if self._use_previous_action else None
        func_action = time_step.observation[self._func_action_key] if self._use_previous_action else None

        assert (self._available_actions_key in time_step.observation) == \
               (self._available_actions_key in self._time_step_spec.observation)
        available_actions = None
        if self._available_actions_key in time_step.observation:
            available_actions = time_step.observation[self._available_actions_key]

        # time_step.observation is a dict of screen, minimap and structured info.
        time_step_obs = time_step.observation
        (spatial_q_values, structured_q_values), q_policy_state = self._mixed_q_network(
            time_step_obs, time_step.step_type, policy_state)
        assert isinstance(spatial_q_values, dict) and isinstance(structured_q_values, dict)
        if available_actions is not None:
            available_actions = tf.convert_to_tensor(available_actions)
            assert all([available_actions.shape == t.shape for t in structured_q_values.values()])
            structured_q_values = tf.nest.map_structure(lambda x: tf.where(tf.equal(available_actions, 1), x, neg_inf),
                                                        structured_q_values)

        # TODO(b/122314058): Validate and enforce that sampling distributions
        # created with the q_network logits generate the right action shapes. This
        # is curretly patching the problem.

        # If the action spec says each action should be shaped (1,), add another
        # dimension so the final shape is (B, 1, A), where A is the number of
        # actions. This will make Categorical emit events shaped (B, 1) rather than
        # (B,). Using axis -2 to allow for (B, T, 1, A) shaped q_values.
        assert all([s.shape.ndims == 1 for s in self._flat_action_spec]) \
               or all([s.shape.ndims == 0 for s in self._flat_action_spec]), \
            "all action specs' ndims should be consistently 1 or 0"

        if previous_action is not None:
             tf.assert_equal(tf.add_n([spatial_q_values[k].shape[-1] if k in spatial_q_values else 0
                                       for k in self._spatial_names]
                                 + [structured_q_values[k].shape[-1] if k in structured_q_values else 0
                                   for k in self._structured_names]),
                        self._func_arg_mask.shape[-1])

        if func_action is None:
            discrete_func_action = None
        else:
            assert isinstance(previous_action, dict)
            discrete_func_action = func_action[self._discrete_action_key] \
                                        if self._discrete_action_key in func_action else None
        spatial_q_values, structured_q_values = self._mask_logits(spatial_q_values, structured_q_values,
                                                                  discrete_func_action)

        if self._flat_action_spec[0].shape.ndims == 1:
            for k, v in spatial_q_values.items():
                spatial_q_values[k] = tf.expand_dims(v, -2)
            for k, v in structured_q_values.items():
                structured_q_values[k] = tf.expand_dims(v, -2)

        # concatenate q_values into single represnetation
        q_values = []
        for name in self._spatial_names:
            if name in spatial_q_values:
                q_values.append(spatial_q_values[name])
        for name in self._structured_names:
            if name in structured_q_values:
                q_values.append(structured_q_values[name])
        q_values = tf.concat(q_values, axis=-1)

        ##TODO: mask_split_fn needs to be investigated
        # logits = spatial_q_values[self._spatial_names[0]]
        # mask_split_fn = self._q_network.mask_split_fn
        #
        # neg_inf = tf.constant(-np.inf, dtype=tf.float32)
        # if mask_split_fn:
        #     _, mask = mask_split_fn(time_step.observation)
        #
        #     # Expand the mask as needed in the same way as q_values above.
        #     if self._flat_action_spec.shape.ndims == 1:
        #         mask = tf.expand_dims(mask, -2)
        #
        #     # Overwrite the logits for invalid actions to -inf.
        #     logits = tf.compat.v2.where(tf.cast(mask, tf.bool), logits, neg_inf)

        # TODO(kbanoop): Handle distributions over nests.
        q_distribution = shifted_categorical.ShiftedCategorical(
            logits=q_values, dtype=tf.int32, shift=0)

        # q_distributions = dict()
        # for k, v in spatial_q_values.items():
        #     print(v.shape)
        #     q_distributions[k] = shifted_categorical.ShiftedCategorical(
        #         logits=v,
        #         dtype=self._action_spec[k].dtype,
        #         shift=self._action_spec[k].minimum[0])
        # for k, v in structured_q_values.items():
        #     print(v.shape)
        #     q_distributions[k] = shifted_categorical.ShiftedCategorical(
        #         logits=v,
        #         dtype=self._action_spec[k].dtype,
        #         shift=self._action_spec[k].minimum[0])
        # q_distribution = tf.nest.pack_sequence_as(self._action_spec, [q_distribution])

        return policy_step.PolicyStep(q_distribution, q_policy_state)

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

        ##TODO: need to consider whether we convert action to (func, arg) here or in actionwrapper.


        return distribution_step._replace(action=actions, info=info)


    def _convert_spatial_action(self, action, height=None, width=None):

        if (height and not width) or (not height and width):
            raise ValueError('height/width should co-exist')

        if height and width:
            action_height = tf.cast(action/width, tf.int32)
            action_width = tf.cast(action - action_height*width, tf.int32)
            discrete_action = tf.cast(action/(height*width), tf.int32)

            return discrete_action, (action_height, action_width)
        else:
            return action

    def _mask_logits(self, spatial_q_values, structured_q_values, previous_action):

        # mask logits based on previous action taken

        neg_inf = tf.constant(-np.inf, dtype=tf.float32)
        if self._func_arg_mask is not None and previous_action is not None:
            assert previous_action.shape.ndims == 1

            previous_action_mask = tf.gather(self._func_arg_mask, previous_action)
            assert previous_action_mask.shape.ndims == 2
            idx_offset = 0
            for name in self._spatial_names:
                if name in spatial_q_values:
                    q_values_shape = spatial_q_values[name].shape
                    height, width, nchannels = q_values_shape[1], q_values_shape[2], q_values_shape[-1]
                    sliced_previous_action_mask = previous_action_mask[:, idx_offset:idx_offset+nchannels]
                    sliced_previous_action_mask = tf.expand_dims(tf.expand_dims(sliced_previous_action_mask, axis=1), axis=1)
                    broadcast_previous_action_mask = tf.tile(sliced_previous_action_mask, [1, height, width, 1])
                    tf.assert_equal(spatial_q_values[name].shape, broadcast_previous_action_mask.shape)

                    spatial_q_values[name] = tf.compat.v2.where(
                        tf.cast(tf.equal(broadcast_previous_action_mask, 1), tf.bool), spatial_q_values[name], neg_inf)
                    spatial_q_values[name] = tf.reshape(spatial_q_values[name], [q_values_shape[0], -1])
                    idx_offset += nchannels

            for name in self._structured_names:
                if name in structured_q_values:
                    nchannels = structured_q_values[name].shape[-1]
                    sliced_previous_action_mask = previous_action_mask[:, idx_offset:idx_offset+nchannels]
                    tf.assert_equal(structured_q_values[name].shape, sliced_previous_action_mask.shape)

                    structured_q_values[name] = tf.compat.v2.where(
                        tf.cast(tf.equal(sliced_previous_action_mask, 1), tf.bool), structured_q_values[name], neg_inf)
                    idx_offset += nchannels

            if idx_offset != self._func_arg_mask.shape[1]:
                raise ValueError("feature num of channels doesn't match func_arg mask shape")
        else:
            for name, v in spatial_q_values.items():
                q_values_shape = v.shape
                spatial_q_values[name] = tf.reshape(v, [q_values_shape[0], -1])

        return spatial_q_values, structured_q_values
