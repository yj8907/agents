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

"""Common utilities for TF-Agents Environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_py_policy
from tf_agents.specs import array_spec


def get_tf_env(environment):
    """Ensures output is a tf_environment, wrapping py_environments if needed."""
    if environment is None:
        raise ValueError('`environment` cannot be None')
    if isinstance(environment, py_environment.PyEnvironment):
        tf_env = tf_py_environment.TFPyEnvironment(environment)
    elif isinstance(environment, tf_environment.TFEnvironment):
        tf_env = environment
    else:
        raise ValueError(
            '`environment` %s must be an instance of '
            '`tf_environment.TFEnvironment` or `py_environment.PyEnvironment`.' %
            environment)
    return tf_env


def validate_py_environment(environment, episodes=5):
    """Validates the environment follows the defined specs."""
    time_step_spec = environment.time_step_spec()
    action_spec = environment.action_spec()

    random_policy = random_py_policy.RandomPyPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec)

    episode_count = 0
    time_step = environment.reset()

    while episode_count < episodes:
        if not array_spec.check_arrays_nest(time_step, time_step_spec):
            raise ValueError(
                'Given `time_step`: %r does not match expected `time_step_spec`: %r' %
                (time_step, time_step_spec))

        action = random_policy.action(time_step).action
        time_step = environment.step(action)

        if time_step.is_last():
            episode_count += 1
            time_step = environment.reset()


def tf_nest_concatenate(nested_tensors, axis=0):
    """
    Implement concatenation of nested tensor objects

    :param nested_tensors: A list of nested tensors.
    :return: concatenated tensors packed as nested_tensors
    """

    assert isinstance(nested_tensors, (list, tuple))
    assert len(nested_tensors) > 0

    flattened_tensors = [tf.nest.flatten(t) for t in nested_tensors]
    concat_flattened_tensors = []

    for tensor in zip(*flattened_tensors):
        assert all([isinstance(t, tf.Tensor) for t in tensor])
        concat_tensor = tf.concat(list(tensor), axis=axis)
        concat_flattened_tensors.append(concat_tensor)

    return tf.nest.pack_sequence_as(nested_tensors[0], concat_flattened_tensors)


def tf_nest_split(nested_tensors, axis=0):
    """
    Implement split of nested tensor objects

    :param nested_tensors: A nested tensors.
           axis:
    :return: concatenated tensors packed as nested_tensors
    """

    flattened_tensors = tf.nest.flatten(nested_tensors)
    assert all([isinstance(t, tf.Tensor) for t in flattened_tensors])

    num_splits = flattened_tensors[0].shape[axis]

    split_flattened_tensors = []
    for tensor in flattened_tensors:
        split_tensors = tf.split(tensor, num_or_size_splits=num_splits, axis=axis)
        split_flattened_tensors.append(split_tensors)

    split_nested_tensors = []
    for tensors in zip(*split_flattened_tensors):
        split_nested_tensors.append(tf.nest.pack_sequence_as(nested_tensors, list(tensors)))

    return split_nested_tensors
