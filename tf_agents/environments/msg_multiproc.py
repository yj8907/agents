import numpy as np
from multiprocessing import Pipe, Process
from tf_agents.environments import env_abc
from tf_agents.environments import utils

START, STEP, RESET, STOP, DONE, CURRENT_TIME_STEP = range(6)


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

    def obs_spec(self):
        return self._env.obs_spec()

    def act_spec(self):
        return self._env.act_spec()

    def time_step_spec(self):
        return self._env.time_step_spec()

    def current_time_step(self):
        self.conn.send((CURRENT_TIME_STEP, None))
        return self.conn.recv()

    def _run(self):
        while True:
            msg, data = self.w_conn.recv()
            if msg == START:
                self._env.start()
                self.w_conn.send(DONE)
            elif msg == CURRENT_TIME_STEP:
                time_step = self._env.current_time_step()
                self.w_conn.send(time_step)
            elif msg == STEP:
                time_step = self._env.step(data)
                self.w_conn.send(time_step)
            elif msg == RESET:
                obs = self._env.reset()
                self.w_conn.send((obs, -1, -1))
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

        :param actions: a dict of batched actions
        :return:
        """
        actions = utils.tf_nest_split(actions)
        for idx, env in enumerate(self.envs):
            env.step(actions[idx])
        return self._observe()

    def reset(self):
        for e in self.envs:
            e.reset()
        return self._observe()

    def _observe(self):
        time_step = self.wait()
        time_step = utils.tf_nest_concatenate(time_step)
        return time_step

    def stop(self):
        for e in self.envs:
            e.stop()
        for e in self.envs:
            e.proc.join()

    def wait(self):
        return [e.wait() for e in self.envs]

    def obs_spec(self):
        return self.envs[0].obs_spec()

    def act_spec(self):
        return self.envs[0].act_spec()

    def time_step_spec(self):
        return self.envs[0].time_step_spec()

    def current_time_step(self):
        return utils.tf_nest_concatenate([e.current_time_step() for e in self.envs])

    @property
    def batch_size(self):
        return len(self.envs)
