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

"""A Driver that takes N steps in the environment using a tf.while_loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from tf_agents.drivers import driver
from tf_agents.policies import discrete_boltzmann_policy
from tf_agents.policies import epsilon_discrete_boltzmann_policy

from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils


@gin.configurable
class DynamicStepDriver(driver.Driver):
  """A driver that takes N steps in an environment using a tf.while_loop.

  The while loop will run num_steps in the environment, only counting steps that
  result in an environment transition, i.e. (time_step, action, next_time_step).
  If a step results in environment resetting, i.e. time_step.is_last() and
  next_time_step.is_first() (traj.is_boundary()), this is not counted toward the
  num_steps.

  As environments run batched time_steps, the counters for all batch elements
  are summed, and execution stops when the total exceeds num_steps. When
  batch_size > 1, there is no guarantee that exactly num_steps are taken -- it
  may be more but never less.

  This termination condition can be overridden in subclasses by implementing the
  self._loop_condition_fn() method.
  """

  def __init__(
      self,
      env,
      policy,
      observers=None,
      transition_observers=None,
      num_steps=1,
  ):
    """Creates a DynamicStepDriver.

    Args:
      env: A tf_environment.Base environment.
      policy: A tf_policy.Base policy.
      observers: A list of observers that are updated after every step in the
        environment. Each observer is a callable(time_step.Trajectory).
      transition_observers: A list of observers that are updated after every
        step in the environment. Each observer is a callable((TimeStep,
        PolicyStep, NextTimeStep)).
      num_steps: The number of steps to take in the environment.

    Raises:
      ValueError:
        If env is not a tf_environment.Base or policy is not an instance of
        tf_policy.Base.
    """
    super(DynamicStepDriver, self).__init__(env, policy, observers,
                                            transition_observers)
    self._num_steps = num_steps
    self._run_fn = common.function_in_tf1()(self._run)

  def _loop_condition_fn(self):
    """Returns a function with the condition needed for tf.while_loop."""

    def loop_cond(counter, *_):
      """Determines when to stop the loop, based on step counter.

      Args:
        counter: Step counters per batch index. Shape [batch_size] when
          batch_size > 1, else shape [].

      Returns:
        tf.bool tensor, shape (), indicating whether while loop should continue.
      """
      return tf.less(tf.reduce_sum(input_tensor=counter), self._num_steps)

    return loop_cond

  def _loop_body_fn(self):
    """Returns a function with the driver's loop body ops."""

    def loop_body(counter, time_step, policy_state):
      """Runs a step in environment.

      While loop will call multiple times.

      Args:
        counter: Step counters per batch index. Shape [batch_size].
        time_step: TimeStep tuple with elements shape [batch_size, ...].
        policy_state: Policy state tensor shape [batch_size, policy_state_dim].
          Pass empty tuple for non-recurrent policies.

      Returns:
        loop_vars for next iteration of tf.while_loop.
      """

      action_step = self.policy.action(time_step, policy_state)
      policy_state = action_step.state
         
      if isinstance(self.policy, (discrete_boltzmann_policy.DiscreteBoltzmannPolicy,
                                  epsilon_discrete_boltzmann_policy.EpsilonDiscreteBoltzmannPolicy)):
          next_time_step = self.env.step(tf.cast(action_step.action[:, 0], tf.int64))
      else:
          next_time_step = self.env.step(action_step.action)

      traj = trajectory.from_transition(time_step, action_step, next_time_step)
      observer_ops = [observer(traj) for observer in self._observers]
      transition_observer_ops = [
          observer((time_step, action_step, next_time_step))
          for observer in self._transition_observers
      ]
      with tf.control_dependencies(
          [tf.group(observer_ops + transition_observer_ops)]):
        time_step, next_time_step, policy_state = tf.nest.map_structure(
            tf.identity, (time_step, next_time_step, policy_state))

      # While loop counter should not be incremented for episode reset steps.
      counter += tf.cast(~traj.is_boundary(), dtype=tf.int32)

      return [counter, next_time_step, policy_state]

    return loop_body

  def run(self, time_step=None, policy_state=None, maximum_iterations=None):
    """Takes steps in the environment using the policy while updating observers.

    Args:
      time_step: optional initial time_step. If None, it will use the
        current_time_step of the environment. Elements should be shape
        [batch_size, ...].
      policy_state: optional initial state for the policy.
      maximum_iterations: Optional maximum number of iterations of the while
        loop to run. If provided, the cond output is AND-ed with an additional
        condition ensuring the number of iterations executed is no greater than
        maximum_iterations.

    Returns:
      time_step: TimeStep named tuple with final observation, reward, etc.
      policy_state: Tensor with final step policy state.
    """
    return self._run_fn(
        time_step=time_step,
        policy_state=policy_state,
        maximum_iterations=maximum_iterations)

  # TODO(b/113529538): Add tests for policy_state.
  def _run(self, time_step=None, policy_state=None, maximum_iterations=None):
    """See `run()` docstring for details."""
    if time_step is None:
      time_step = self.env.current_time_step()
    if policy_state is None:
      policy_state = self.policy.get_initial_state(self.env.batch_size)

    # Batch dim should be first index of tensors during data collection.
    batch_dims = nest_utils.get_outer_shape(time_step,
                                            self.env.time_step_spec())
    counter = tf.zeros(batch_dims, tf.int32)

    [_, time_step, policy_state] = tf.while_loop(
        cond=self._loop_condition_fn(),
        body=self._loop_body_fn(),
        loop_vars=[counter, time_step, policy_state],
        back_prop=False,
        parallel_iterations=1,
        maximum_iterations=maximum_iterations,
        name='driver_loop')
    return time_step, policy_state
