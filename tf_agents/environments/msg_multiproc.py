import numpy as np
from multiprocessing import Pipe, Process
from tf_agents.environments import env_abc

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import protocol
from pysc2.env.environment import StepType
from absl import flags
from pysc2.env import sc2_env

START, STEP, RESET, STOP, DONE = range(5)


class BaesSC2Env(env_abc.Env):

    def __init__(
            self,
            map_name='MoveToBeacon',
            render=False,
            reset_done=True,
            max_ep_len=None,
            screen_dim=24,
            minimap_dim=16,
            step_mul=8,
    ):
        super(BaesSC2Env, self).__init__(map_name, render, reset_done, max_ep_len)

        self.step_mul = step_mul
        self._screen_dim = screen_dim
        self._minimap_dim = minimap_dim
        self._env = None

    def start(self):
        # importing here to lazy-load

        # fail-safe if executed not as absl app
        if not flags.FLAGS.is_parsed():
            flags.FLAGS([""])
        #            flags.FLAGS(sys.argv)
        try:
            self._env = sc2_env.SC2Env(
                map_name=self.id,
                visualize=self.render,
                agent_interface_format=[features.parse_agent_interface_format(
                    feature_screen=self._screen_dim,
                    feature_minimap=self._minimap_dim,
                    rgb_screen=None,
                    rgb_minimap=None
                )],
                step_mul=self.step_mul, )
        except protocol.ConnectionError:
            self.start()

    def step(self, action):
        try:
            time_step = self._env.step(action)
        except protocol.ConnectionError:
            self.restart()
            return self.reset(), 0, 1

        if time_step[0].step_type == StepType.LAST and self.reset_done:
            reset_timestep = self.reset()
            time_step[0]._replace(observation=reset_timestep.observation)

        return time_step

    def reset(self):
        try:
            time_step = self._env.reset()
            time_step = self._env.reset()
        except protocol.ConnectionError:
            self.restart()
            return self.reset()
        return time_step

    def stop(self):
        self._env.close()

    def restart(self):
        self.stop()
        self.start()


class MsgProcEnv(env_abc.Env):
    def __init__(self, env):
        super().__init__(env.id)
        self._env = env
        self.conn = self.w_conn = self.proc = None

    def start(self):
        self.conn, self.w_conn = Pipe()
        self.proc = Process(target=self._run)
        self.proc.start()
        self.conn.send((START, None))

    def step(self, act):
        self.conn.send((STEP, act))

    def reset(self):
        self.conn.send((RESET, None))

    def stop(self):
        self.conn.send((STOP, None))

    def wait(self):
        return self.conn.recv()

    def _run(self):
        while True:
            msg, data = self.w_conn.recv()
            if msg == START:
                self._env.start()
                self.w_conn.send(DONE)
            elif msg == STEP:
                time_step = self._env.step(data)
                self.w_conn.send(time_step)
            elif msg == RESET:
                obs = self._env.reset()
                self.w_conn.send(obs)
            elif msg == STOP:
                self._env.stop()
                self.w_conn.close()
                break

class MsgMultiProcEnv(env_abc.Env):
    """
    Parallel environments via multiprocessing + pipes
    """
    def __init__(self, envs):
        super().__init__(envs[0].id)
        self.envs = [MsgProcEnv(env) for env in envs]

    def start(self):
        for env in self.envs:
            env.start()
        self.wait()

    def step(self, actions):
        """

        :param actions: a list of Function Calls that can be passed directly to sc2 envs
        :return:
        """
        for idx, env in enumerate(self.envs):
            env.step(actions[idx])
        return self._observe()

    def reset(self):
        for e in self.envs:
            e.reset()
        return self._observe()

    def _observe(self):
        return self.wait()

    def stop(self):
        for e in self.envs:
            e.stop()
        for e in self.envs:
            e.proc.join()

    def wait(self):
        return [e.wait() for e in self.envs]
