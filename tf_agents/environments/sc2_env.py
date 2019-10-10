import sys
import gin
import numpy as np
from absl import flags
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import protocol
from pysc2.env.environment import StepType

from tf_agents.environments import env_abc
from tf_agents.environments import msg_multiproc
from tf_agents.environments import utils
from tf_agents.networks import mixed_q_network
from tf_agents.trajectories import time_step as ts

import tensorflow as tf
import collections
import copy

from tensorflow.python.framework import tensor_spec

ACTIONS_MINIGAMES, ACTIONS_MINIGAMES_ALL, ACTIONS_ALL = ['minigames', 'minigames_all', 'all']


@gin.configurable
class SC2EnvWrapper(env_abc.Env):
    """
    'minigames' action set is enough to solve all minigames listed in SC2LE
    'minigames_all' expands that set with actions that may improve end results, but will drop performance
    'all' is the full action set, only necessary for generic agent playing full game with all three races

    You can also specify your own action set in the gin config file under SC2Env.action_ids
    Full list of available actions https://github.com/deepmind/pysc2/blob/master/pysc2/lib/actions.py#L447-L1008
    """

    def __init__(
            self,
            map_name='MoveToBeacon',
            render=False,
            reset_done=True,
            max_ep_len=None,
            screen_dim=24,
            minimap_dim=16,
            step_mul=16,
            batch_size=1,
            obs_features=None,
            action_ids=ACTIONS_MINIGAMES
    ):
        super(SC2EnvWrapper, self).__init__(map_name, render, reset_done, max_ep_len)

        self.step_mul = step_mul
        self._screen_dim = screen_dim
        self._minimap_dim = minimap_dim
        self._env = None
        self.batch_size = batch_size
        self._state = None

        # sensible action set for all minigames
        if not action_ids or action_ids in [ACTIONS_MINIGAMES, ACTIONS_MINIGAMES_ALL]:
            action_ids = [0, 1, 2, 3, 4, 6, 7, 12, 13, 42, 44, 50, 91, 183, 234, 309, 331, 332, 333, 334, 451, 452, 490]

        # some additional actions for minigames (not necessary to solve)
        if action_ids == ACTIONS_MINIGAMES_ALL:
            action_ids += [11, 71, 72, 73, 74, 79, 140, 168, 239, 261, 264, 269, 274, 318, 335, 336, 453, 477]

        # full action space, including outdated / unusable to current race / usable only in certain cases
        if action_ids == ACTIONS_ALL:
            action_ids = [f.id for f in actions.FUNCTIONS]

        # by default use majority of obs features, except for some that are unnecessary for minigames
        # e.g. race-specific like creep and shields or redundant like player_id
        if not obs_features:
            obs_features = {
                'screen': ['player_relative', 'selected', 'visibility_map', 'unit_hit_points_ratio', 'unit_density'],
                'minimap': ['player_relative', 'selected', 'visibility_map', 'camera'],
                # available actions should always be present and in first position
                'non-spatial': ['available_actions', 'player']}

        self.act_wrapper = ActionWrapper(screen_dim, action_ids)
        self.obs_wrapper = ObservationWrapper(obs_features, action_ids)
        self.input_tensor_specs, self.action_specs, self.func_masks = None, None, None
        self._time_step_spec = None

    @property
    def batched(self):
        return True

    def start(self):
        # importing here to lazy-load
        from pysc2.env import sc2_env

        # fail-safe if executed not as absl app
        if not flags.FLAGS.is_parsed():
            flags.FLAGS([""])
#            flags.FLAGS(sys.argv)

        sc2_envs = []
        for i in range(self.batch_size):
            env = msg_multiproc.BaesSC2Env(map_name=self.id,
                render=self.render,
                reset_done=self.reset_done,
                max_ep_len=self.max_ep_len,
                screen_dim=self._screen_dim,
                minimap_dim=self._minimap_dim,
                step_mul=self.step_mul,)
            sc2_envs.append(env)
        sc2_multiproc_env = msg_multiproc.MsgMultiProcEnv(sc2_envs)

        self._env = sc2_multiproc_env
        self._env.start()

    def _wrap_time_step(self, time_steps):

        assert isinstance(time_steps, (list, tuple))
        time_steps = list(zip(*[self.obs_wrapper(ts) for ts in time_steps]))
        step_type, reward, discount, obs_wrapped = [utils.tf_nest_concatenate(self._batch_tensors(ts))
                                                    for ts in time_steps]

        flat_observations = tf.nest.flatten(obs_wrapped)
        time_step = self._set_names_and_shapes(step_type, reward, discount, *flat_observations)

        return time_step

    def step(self, action):

        split_actions = utils.tf_nest_split(action, axis=0)
        split_actions = [self.act_wrapper(act) for act in split_actions]

        time_steps = self._env.step(split_actions)
        wrapped_time_step = self._wrap_time_step(time_steps)

        # handle sc2 step_type.LAST
        if self._state is not None:
            reset_step_type = tf.where(self._state, StepType.FIRST.value, wrapped_time_step.step_type)
            wrapped_time_step = wrapped_time_step._replace(step_type=reset_step_type)

        self._state = wrapped_time_step.step_type == StepType.LAST

        return wrapped_time_step

    def _reset(self):
        time_steps = self._env.reset()
        return self._wrap_time_step(time_steps)

    def reset(self):

        time_step = self._reset()
        return time_step

    def stop(self):
        self._env.stop()

    def restart(self):
        self.stop()
        self.start()

    def obs_spec(self):
        if not self.obs_wrapper.obs_spec:
            self.make_specs()
        return self.obs_wrapper.obs_spec

    def act_spec(self):
        if not self.act_wrapper.spec:
            self.make_specs()
        return self.act_wrapper.spec

    def time_step_spec(self):

        if self._time_step_spec is None:
            raise ValueError("run make_specs first")
        else:
            return self._time_step_spec

    def current_time_step(self):
        """Returns the current ts.TimeStep.

        Returns:
          A `TimeStep` tuple of:
            step_type: A scalar int32 tensor representing the `StepType` value.
            reward: A scalar float32 tensor representing the reward at this
              timestep.
            discount: A scalar float32 tensor representing the discount [0, 1].
            observation: A Tensor, or a nested dict, list or tuple of Tensors
              corresponding to `observation_spec()`.
        """

        with tf.name_scope('current_time_step'):
            return self.reset()

    def _batch_tensors(self, time_step):
        ##TODO: implement multiprocessing
        """
        Temporary fix to add batch dimension
        :param time_step: ts.time_step object
        :return:
        """
        return tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), time_step)

    def _set_names_and_shapes(self, step_type, reward, discount,
                              *flat_observations):
        """Returns a `TimeStep` namedtuple."""
        step_type = tf.identity(step_type, name='step_type')
        reward = tf.identity(reward, name='reward')
        discount = tf.identity(discount, name='discount')
        batch_shape = () if not self.batched else (self.batch_size,)
        batch_shape = tf.TensorShape(batch_shape)
        if not tf.executing_eagerly():
            # Shapes are not required in eager mode.
            reward.set_shape(batch_shape)
            step_type.set_shape(batch_shape)
            discount.set_shape(batch_shape)

        # Give each tensor a meaningful name and set the static shape.
        named_observations = []
        for obs, spec in zip(flat_observations,
                             tf.nest.flatten(self.obs_spec())):
            named_observation = tf.identity(obs, name=spec.name)
            named_observation = tf.cast(named_observation, spec.dtype)
            if not tf.executing_eagerly():
                named_observation.set_shape(batch_shape.concatenate(spec.shape))
            named_observations.append(named_observation)

        observations = tf.nest.pack_sequence_as(self.obs_spec(),
                                                named_observations)
        time_step = ts.TimeStep(step_type, reward, discount, observations)
        time_step = tf.nest.map_structure(lambda x, t: tf.cast(x, t.dtype), time_step, self._time_step_spec)

        return time_step

    def extract_action_spec(self, step):

        action_specs = [self.act_wrapper._merged_spatial_action_spec,
                        self.act_wrapper._merged_structured_action_spec]

        if step < -1:
            print("generate None spec for step < -1")
            return None

        action_spec = dict()
        if step >= 0:
            for spec in action_specs:
                for k, v in spec.items():
                    if step < len(v):
                        action_spec[k] = v[step]
        else:
            action_spec['structured'] = tensor_spec.BoundedTensorSpec(
                shape=(), dtype=np.int32, name="sc2_func_action_spec",
                minimum=(0,),
                maximum=(len(self.act_wrapper.func_ids) - 1,))

        return action_spec

    def extract_previous_action_spec(self, current_action_step):

        previous_action_spec = None
        if current_action_step == -1:
            previous_action_spec = None
        elif current_action_step == 0:
            previous_action_spec = dict()
            previous_action_spec['discrete'] = tensor_spec.BoundedTensorSpec(
                shape=(), dtype=np.int32,
                name='functions',
                minimum=(0,), maximum=(len(self.act_wrapper.func_ids) - 1,))
        elif current_action_step == 1:
            action_specs = self.extract_action_spec(current_action_step - 1)
            num_discrete_actions = sum([v.maximum[0] - v.minimum[0] + 1
                                        for k, v in action_specs.items()])

            previous_action_spec = dict()
            previous_action_spec['discrete'] = tensor_spec.BoundedTensorSpec(
                shape=(), dtype=np.int32,
                name='discrete_func',
                minimum=(0,), maximum=(num_discrete_actions - 1,))
            previous_action_spec['continuous'] = tensor_spec.BoundedTensorSpec(
                shape=(2,), dtype=np.float32,
                name='continuous_func',
                minimum=(-np.inf,), maximum=(np.inf,))

        return previous_action_spec

    def extract_func_action_spec(self, current_action_step):

        if current_action_step == -1:
            func_action_spec = None
        else:
            func_action_spec = dict()
            func_action_spec['discrete'] = tensor_spec.BoundedTensorSpec(
                shape=(), dtype=np.int32,
                name='functions',
                minimum=(0,), maximum=(len(self.act_wrapper.func_ids) - 1,))

        return func_action_spec

    def arg_spec_generator(self, action_step):

        action_specs = self.extract_action_spec(action_step)

        input_tensor_specs = copy.deepcopy(self.obs_wrapper.obs_spec)
        input_tensor_specs['previous_action'] = self.extract_previous_action_spec(action_step)
        input_tensor_specs['func_action'] = self.extract_func_action_spec(action_step)

        if input_tensor_specs['previous_action'] is None:
            input_tensor_specs.pop('previous_action')
        if input_tensor_specs['func_action'] is None:
            input_tensor_specs.pop('func_action')

        if action_step >= 0:
            func_action_mask = self.act_wrapper.func_action_mask[action_step]
            input_tensor_specs.pop('available_actions')
        else:
            func_action_mask = None

        return list2tuple(input_tensor_specs), action_specs, func_action_mask

    def make_specs(self):
        # importing here to lazy-load
        from pysc2.env import mock_sc2_env
        mock_env = mock_sc2_env.SC2TestEnv(map_name=self.id, agent_interface_format=[
            features.parse_agent_interface_format(feature_screen=self._screen_dim, feature_minimap=self._minimap_dim)])
        self.act_wrapper.make_spec(mock_env.action_spec(), mock_env.observation_spec())
        self.obs_wrapper.make_spec(mock_env.observation_spec())

        self.input_tensor_specs = []
        self.action_specs = []
        self.func_masks = []

        action_steps = [-1, 0, 1]
        for action_step in action_steps:
            input_spec, action_spec, func_action_mask = self.arg_spec_generator(action_step)
            self.input_tensor_specs.append(input_spec)
            self.action_specs.append(action_spec)
            self.func_masks.append(func_action_mask)
        mock_env.close()

        self._time_step_spec = ts.time_step_spec(self.obs_wrapper.obs_spec)

        return

    def generate_mock_modeldata(self, input_specs, action_specs):
        q_networks = []

        action_steps = [-1, 0, 1]
        assert len(action_steps) == len(input_specs) == len(action_specs)
        for action_step, input_spec, action_spec in zip(action_steps, input_specs, action_specs):
            with tf.compat.v1.variable_scope('step'+str(action_step+1), reuse=tf.compat.v1.AUTO_REUSE) as scope:
                q_net = mixed_q_network.MixedQNetwork(input_spec, action_spec,
                                                      name='MixedQNetwork' + str(action_step + 1))
                q_networks.append(q_net)
        return q_networks

    def observation_generator(self, action_step, action_spec, batch_size=1):

        input_tensor_spec = self.input_tensor_specs[action_step+1]

        screen_image = tf.ones(
            (batch_size,) + input_tensor_spec['screen'][0].shape + (len(input_tensor_spec['screen']),),
            tf.float32)
        minimap_image = tf.ones(
            (batch_size,) + input_tensor_spec['minimap'][0].shape + (len(input_tensor_spec['minimap']),),
            tf.float32)

        observation = dict()
        observation['screen'] = tf.unstack(screen_image, len(input_tensor_spec['screen']), axis=-1)
        observation['minimap'] = tf.unstack(minimap_image, len(input_tensor_spec['minimap']), axis=-1)

        for i in range(len(input_tensor_spec['screen'])):
            observation['screen'][i] = tf.cast(observation['screen'][i],
                                               input_tensor_spec['screen'][i].dtype)
        for i in range(len(input_tensor_spec['minimap'])):
            observation['minimap'][i] = tf.cast(observation['minimap'][i],
                                                input_tensor_spec['minimap'][i].dtype)

        previous_action = None
        if action_step >= 0:
            previous_action = dict()
            for k, v in action_spec.items():
                if k == 'discrete':
                    previous_action[k] = tf.ones((batch_size,), np.int32) * 1
                if k == 'continuous':
                    previous_action[k] = tf.ones((batch_size, v.shape[0]), np.float32) * 1
        if action_step == -1:
            action_dim = action_spec['structured'].maximum[0] - action_spec['structured'].minimum[0] + 1
            observation['available_actions'] = np.random.randint(0, 2, (batch_size, action_dim))

        return observation, previous_action


class ObservationWrapper:
    def __init__(self, _features=None, action_ids=None):
        self.obs_spec = dict()
        self.features = _features
        self.action_ids = action_ids

        screen_feature_to_idx = {feat: idx for idx, feat in enumerate(features.SCREEN_FEATURES._fields)}
        minimap_feature_to_idx = {feat: idx for idx, feat in enumerate(features.MINIMAP_FEATURES._fields)}

        self.feature_masks = {
            'screen': [screen_feature_to_idx[f] for f in _features['screen']],
            'minimap': [minimap_feature_to_idx[f] for f in _features['minimap']]
        }

    def __call__(self, timestep):
        ts = timestep[0]
        step_type, reward, discount, obs = ts.step_type, ts.reward, ts.discount, ts.observation

        obs_wrapped = dict()
        obs_wrapped['screen'] = tf.unstack(obs['feature_screen'][self.feature_masks['screen']],
                                            len(self.feature_masks['screen']), axis=0)
        obs_wrapped['minimap'] = tf.unstack(obs['feature_minimap'][self.feature_masks['minimap']],
                                            len(self.feature_masks['minimap']), axis=0)

        for feat_name in self.features['non-spatial']:
            if feat_name == 'available_actions':
                fn_ids_idxs = [i for i, fn_id in enumerate(self.action_ids) if fn_id in obs[feat_name]]
                mask = np.zeros((len(self.action_ids),), dtype=np.int32)
                mask[fn_ids_idxs] = 1
                obs[feat_name] = mask
                obs_wrapped['available_actions'] = obs[feat_name]

        return step_type, reward, discount, obs_wrapped

    def make_spec(self, spec):
        spec = spec[0]

        default_dims = {
            'available_actions': (len(self.action_ids),),
        }

        #        screen_shape = tuple([len(self.features['screen'])]+list(spec['feature_screen'][1:]))
        #        minimap_shape = tuple([len(self.features['minimap'])]+list(spec['feature_minimap'][1:]))

        # since all screen variables are integers, shapes are (screen_size, screen_size) instead of
        # (1, screen_size, screen_size)
        screen_shape = list(spec['feature_screen'][1:])
        minimap_shape = list(spec['feature_minimap'][1:])

        screen_dims = get_spatial_dims(self.features['screen'], features.SCREEN_FEATURES)
        minimap_dims = get_spatial_dims(self.features['minimap'], features.MINIMAP_FEATURES)
        screen_types = get_spatial_type(self.features['screen'], features.SCREEN_FEATURES)
        minimap_types = get_spatial_type(self.features['minimap'], features.MINIMAP_FEATURES)

        obs_spec = dict()
        obs_spec['screen'] = []
        obs_spec['minimap'] = []

        for feat_name, screen_dim, screen_type in zip(self.features['screen'], screen_dims, screen_types):
            action_dtype = np.int32 if screen_type == "CATEGORICAL" else np.float32
            spec = tensor_spec.BoundedTensorSpec(
                    shape=screen_shape, dtype=action_dtype,
                    name='screen_'+feat_name,
                    minimum=(0,), maximum=(screen_dim - 1,))
            obs_spec['screen'].append(spec)

        for feat_name, minimap_dim, minimap_type in zip(self.features['minimap'], minimap_dims, minimap_types):
            action_dtype = np.int32 if minimap_type == "CATEGORICAL" else np.float32
            spec = tensor_spec.BoundedTensorSpec(
                    shape=minimap_shape, dtype=action_dtype,
                    name='minimap_'+feat_name,
                    minimum=(0,), maximum=(minimap_dim - 1,))
            obs_spec['minimap'].append(spec)
        obs_spec['available_actions'] = tensor_spec.BoundedTensorSpec(
                    shape=default_dims['available_actions'], dtype=np.int32,
                    name='available_actions',
                    minimum=(0,), maximum=(1,))

        obs_spec = list2tuple(obs_spec)
        ##TODO: implement structural observation specs

        self.obs_spec = obs_spec


class ActionWrapper:
    def __init__(self, spatial_dim, action_ids, args=None):
        self.spec = None
        if not args:
            args = [
                'screen',
                'minimap',
                'screen2',
                'queued',
                'control_group_act',
                'control_group_id',
                'select_add',
                'select_point_act',
                'select_unit_act',
                # 'select_unit_id'
                'select_worker',
                'build_queue_id',
                # 'unload_id'
            ]
        self.func_ids = action_ids
        self.args, self.spatial_dim = args, spatial_dim

        self._spatial_func_args, self._structured_func_args, self._func_ids_spec = None, None, None
        self._merged_structured_action_spec, self._merged_spatial_action_spec = None, None
        self._spatial_spec = dict()
        self._spatial_id2func, self._spatial_func2id = dict(), dict()
        self._structured_id2func, self._structured_func2id = dict(), dict()
        self._func_action_mask = None
        self._act2func = {}
        self._stacked_id2fuc = dict()
        self._num_actions_per_step = dict()

        self._spatial_action_types, self._structured_action_types = ['screen', 'minimap'], ['structured']
        self._spatial_arg_names = ['screen', 'screen2', 'minimap']
        self._obs_spec = None

    def __call__(self, action):
        """
        :param action: A dict of 'raw' action and 'transformed' action.
            'raw' action is a Tensor of size (3,), 'transformed' action is a Tensor of size (3, 3)
            'transformed action' is used to drive sc2 environment.
        the first dimension represents discrete action and the remaining dimensions represent continuous actions.
        :return:
        """

        defaults = {
            'control_group_act': 0,
            'control_group_id': 0,
            'select_point_act': 0,
            'select_unit_act': 0,
            'select_unit_id': 0,
            'build_queue_id': 0,
            'unload_id': 0,
        }
        action = action['transformed']
        # handle batched action
        if len(action.shape) == 3:
            assert action.shape[0] == 1
            action = np.squeeze(action, axis=0)

        action = copy.deepcopy(action)
        fn_id_idx = int(action[0][0])
        if fn_id_idx >= len(self.func_ids):
            print(action)
            raise ValueError
        fn_id = self.func_ids[fn_id_idx]

        action_type = actions.FUNCTIONS[fn_id]
        action_args = action_type.args

        total_num_steps = 3
        args = []
        arg_types = []
        for step in range(1, total_num_steps):
            # action wrapper step is shifted left with regard to action input
            if action[step][0] > self._num_actions_per_step[step-1]:
                raise ValueError('action index is larger than max number of actions')
            elif action[step][0] < self._num_actions_per_step[step-1]:
                input_arg_type = self._stacked_id2fuc[step-1][int(action[step][0])]

                if input_arg_type[1][0].name not in self._spatial_arg_names:
                    args.append([input_arg_type[1][1]])
                    arg_types.append(input_arg_type[1][0].name)

                input_spatial_args = list(action[step][1:])
                if input_arg_type[0] in self._spatial_arg_names:
                    # convert ratio to absolute values
                    if input_arg_type[0] in ['screen', 'screen2']:
                        if step < 2:
                            arg_types.append('screen')
                        else:
                            arg_types.append('screen2')
                        sizes = self._obs_spec['feature_screen'][1:]
                        assert len(input_spatial_args) == len(sizes)
                        input_spatial_args = [int(a*b) for a, b in zip(input_spatial_args, sizes)]
                    else:
                        arg_types.append('minimap')
                        sizes = self._obs_spec['feature_minimap'][1:]
                        assert len(input_spatial_args) == len(sizes)
                        input_spatial_args = [int(a*b) for a, b in zip(input_spatial_args, sizes)]
                    args.append(input_spatial_args)

        if not len(action_args) <= len(args) == len(arg_types):
            print(action)
            print(action_args)
            print(args)
            raise ValueError("args don't match")
        args = args[:len(action_args)]
        arg_types = arg_types[:len(action_args)]
        if not all([a == b.name for a, b in zip(arg_types, action_args)]):
            print(arg_types)
            print(action_args)
            raise ValueError("arg_types don't match pysc2 action arguments")

        return [actions.FunctionCall(fn_id, args)]

    @property
    def func_action_mask(self):
        return self._func_action_mask

    def _stack_id2func(self):

        max_steps = 0
        for k, v in self._spatial_id2func.items():
            max_steps = max(max_steps, max(list(v.keys())))
        for k, v in self._structured_id2func.items():
            max_steps = max(max_steps, max(list(v.keys())))

        for step in range(max_steps+1):
            self._stacked_id2fuc[step] = dict()
            idx = 0
            for type_name in self._spatial_action_types:
                if step in self._spatial_id2func[type_name]:
                    for v in self._spatial_id2func[type_name][step].values():
                        self._stacked_id2fuc[step][idx] = (type_name, v)
                        idx += 1
            for type_name in self._structured_action_types:
                if step in self._structured_id2func[type_name]:
                    for v in self._structured_id2func[type_name][step].values():
                        self._stacked_id2fuc[step][idx] = (type_name, v)
                        idx += 1
            self._num_actions_per_step[step] = idx

        return

    def make_spec(self, spec, obs_spec):
        spec = spec[0]
        self._obs_spec = obs_spec[0]

        ## continous parameter space: screen, minimap, screen2,
        for t in self._spatial_action_types:
            args = getattr(spec.types, t)
            self._spatial_spec[t] = tensor_spec.BoundedTensorSpec(
                shape=args.sizes, dtype=np.int32, name=t,
                minimum=(0,),
                maximum=(255,))

        structured_func_args = collections.defaultdict(list)
        spatial_func_args = collections.defaultdict(list)

        for fn_id in self.func_ids:
            sc2_func = actions.FUNCTIONS[fn_id]

            check_screen = any([arg.name is 'screen' for arg in sc2_func.args])
            check_minimap = any([arg.name is 'minimap' for arg in sc2_func.args])
            if check_screen and check_minimap:
                raise ValueError('one action can''t act on both screen and minimap')

            current_structured_args = []
            current_spatial_args = []

            if any([check_screen, check_minimap]):
                spatial_type = 'screen' if check_screen else 'minimap'
                for arg in sc2_func.args:
                    if arg.name not in self._spatial_action_types:
                        current_spatial_args.append(arg)
                    elif len(sc2_func.args) == 1:
                        current_spatial_args.append(arg)
                spatial_func_args[spatial_type].append((fn_id, current_spatial_args))
            else:
                for arg in sc2_func.args:
                    current_structured_args.append(arg)
                structured_func_args['structured'].append((fn_id, current_structured_args))
        self._spatial_func_args = spatial_func_args
        self._structured_func_args = structured_func_args

        self.build_spatial_action_spec()
        self.build_structured_action_spec()
        self.build_action_arg_mask()
        self._stack_id2func()
        self._func_ids_spec = tensor_spec.BoundedTensorSpec(
                    shape=(1,), dtype=np.int32,
                    name='functions',
                    minimum=0, maximum=len(self.func_ids) - 1)

    def build_spatial_action_spec(self):

        merged_spatial_action_spec = collections.defaultdict(list)
        for action_type, func_args in self._spatial_func_args.items():
            max_num_args = max([len(arg[1]) for arg in func_args])
            args_per_step = collections.defaultdict(set)

            action_shape = (1,)
            action_dtype = tf.int32

            for args in func_args:
                for i, arg in enumerate(args[1]):
                    args_per_step[i].add(arg)

            idx2arg, arg2idx = dict(), dict()
            for step_i in range(max_num_args):
                idx2arg[step_i], arg2idx[step_i] = dict(), dict()
                args_per_step[step_i] = list(args_per_step[step_i])
                idx = 0
                for arg in args_per_step[step_i]:
                    if arg.name not in ['screen2', 'minimap']:
                        assert len(arg.sizes) == 1
                        arg2idx[step_i][arg] = (idx, idx+arg.sizes[0])
                        for j in range(arg.sizes[0]):
                            idx2arg[step_i][idx] = (arg, j)
                            idx += 1
                    else:
                        arg2idx[step_i][arg] = (idx, idx+1)
                        idx2arg[step_i][idx] = (arg, 0)
                        idx += 1
                combined_spec = tensor_spec.BoundedTensorSpec(
                    shape=action_shape, dtype=action_dtype,
                    name=action_type + '_step_' + str(step_i),
                    minimum=(0,), maximum=(idx - 1,))
                merged_spatial_action_spec[action_type].append(combined_spec)
            self._spatial_id2func[action_type] = idx2arg
            self._spatial_func2id[action_type] = arg2idx
        self._merged_spatial_action_spec = list2tuple(merged_spatial_action_spec)

        return

    def build_structured_action_spec(self):

        merged_structured_action_spec = collections.defaultdict(list)
        for action_type, func_args in self._structured_func_args.items():
            max_num_args = max([len(arg[1]) for arg in func_args])
            args_per_step = collections.defaultdict(set)

            action_shape = (1,)
            action_dtype = tf.int32

            for args in func_args:
                for i, arg in enumerate(args[1]):
                    args_per_step[i].add(arg)

            idx2arg, arg2idx = dict(), dict()
            for step_i in range(max_num_args):
                idx2arg[step_i], arg2idx[step_i] = dict(), dict()
                args_per_step[step_i] = list(args_per_step[step_i])
                idx = 0
                for arg in args_per_step[step_i]:
                    if arg.name is not 'screen2':
                        assert len(arg.sizes) == 1
                        arg2idx[step_i][arg] = (idx, idx+arg.sizes[0])
                        for j in range(arg.sizes[0]):
                            idx2arg[step_i][idx] = (arg, j)
                            idx += 1
                    else:
                        arg2idx[step_i][arg] = (idx, idx+1)
                        idx2arg[step_i][idx] = (arg, 0)
                        idx += 1
                combined_spec = tensor_spec.BoundedTensorSpec(
                    shape=action_shape, dtype=action_dtype,
                    name=action_type + '_step_' + str(step_i),
                    minimum=(0,), maximum=(idx - 1,))
                merged_structured_action_spec[action_type].append(combined_spec)
            self._structured_id2func[action_type] = idx2arg
            self._structured_func2id[action_type] = arg2idx
        self._merged_structured_action_spec = list2tuple(merged_structured_action_spec)

        return

    def build_action_arg_mask(self, spatial_names=("screen", "minimap"), structured_names=("structured",)):

        """
        self._merged_spatial_action_spec: A dict of spatial specs, key: action type, values: A list of merge tensorspec
        self._merged_structured_action_spec: A dict of structural specs, key: action type, values: A list of merge tensorspec
        action_arg_mask: A list of function-action mask for each step.
        :return:
        """

        max_steps = 0
        for v in self._merged_spatial_action_spec.values():
            max_steps = max(max_steps, len(v))
        for v in self._merged_structured_action_spec.values():
            max_steps = max(max_steps, len(v))

        # create mask with zeros
        # spec size in spec_size_per_step is the actual final action spec size
        spec_size_per_step = [0 for i in range(max_steps)]
        for step_i in range(max_steps):
            for type in spatial_names:
                assert type in self._merged_spatial_action_spec
                v = self._merged_spatial_action_spec[type]
                if step_i < len(v):
                    spec_size_per_step[step_i] += v[step_i].maximum[0] - v[step_i].minimum[0] + 1
            for type in structured_names:
                assert type in self._merged_structured_action_spec
                v = self._merged_structured_action_spec[type]
                if step_i < len(v):
                    spec_size_per_step[step_i] += v[step_i].maximum[0] - v[step_i].minimum[0] + 1

        # fill mask with ones
        # the order by which we iterate through the specs should be consistent
        # across ActionWrapper and train agent.
        action_arg_mask = dict()
        for step_i, s in enumerate(spec_size_per_step):
            action_arg_mask[step_i] = np.zeros((len(self.func_ids), s))

            idx_offset = 0
            max_mask_end_idx = 0
            for action_type in spatial_names:
                assert action_type in self._merged_spatial_action_spec
                v = self._merged_spatial_action_spec[action_type]
                for func_id, args in self._spatial_func_args[action_type]:
                    if step_i < len(args):
                        arg = args[step_i]
                        func_idx = self.func_ids.index(func_id)
                        mask_start_idx = idx_offset + self._spatial_func2id[action_type][step_i][arg][0]
                        mask_end_idx = idx_offset + self._spatial_func2id[action_type][step_i][arg][1]
                        if mask_end_idx > s:
                            raise ValueError("mask_end_idx > mask.shape[1]")
                        action_arg_mask[step_i][func_idx, mask_start_idx: mask_end_idx] = 1.0
                if step_i < len(v):
                    idx_offset += v[step_i].maximum[0] - v[step_i].minimum[0] + 1

            for action_type in structured_names:
                assert action_type in self._merged_structured_action_spec
                v = self._merged_structured_action_spec[action_type]
                for func_id, args in self._structured_func_args[action_type]:
                    if step_i < len(args):
                        arg = args[step_i]
                        func_idx = self.func_ids.index(func_id)
                        mask_start_idx = idx_offset + self._structured_func2id[action_type][step_i][arg][0]
                        mask_end_idx = idx_offset + self._structured_func2id[action_type][step_i][arg][1]
                        max_mask_end_idx = max(max_mask_end_idx, mask_end_idx)
                        if mask_end_idx > s:
                            raise ValueError("mask_end_idx > mask.shape[1]")
                        action_arg_mask[step_i][func_idx, mask_start_idx: mask_end_idx] = 1.0
                if step_i < len(v):
                    idx_offset += v[step_i].maximum[0] - v[step_i].minimum[0] + 1
            if max_mask_end_idx != idx_offset:
                raise ValueError("mask end idx should equal mask.shape[1]")
        self._func_action_mask = action_arg_mask

        return

def get_spatial_dims(feat_names, feats):
    feats_dims = []
    for feat_name in feat_names:
        feat = getattr(feats, feat_name)
        feats_dims.append(1)
        if feat.type == features.FeatureType.CATEGORICAL:
            feats_dims[-1] = feat.scale
    return feats_dims


def get_spatial_type(feat_names, feats):
    feats_types = []
    for feat_name in feat_names:
        feat = getattr(feats, feat_name)
        feats_types.append("CATEGORICAL")
        if feat.type != features.FeatureType.CATEGORICAL:
            feats_types[-1] = "SCALA"
    return feats_types


def list2tuple(spec):
    """
    tf_agents assert spec should be tuple instead of list
    :param spec: TensorSpec object
    :return:
    """
    ##TODO: should implement nested list2tuple
    if isinstance(spec, dict):
        for k, v in spec.items():
            if isinstance(v, list):
                spec[k] = tuple(v)
    return spec


