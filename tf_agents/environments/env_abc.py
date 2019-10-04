from abc import ABCMeta, abstractmethod
# from reaver.utils.typing import *


class Env(object):
    __metaclass__ = ABCMeta
    """
    Abstract Base Class for all environments supported by Reaver
    Acts as a glue between the agents, models and envs modules
    Implementing class can be a simple wrapper (e.g. over openAI Gym)

    Note: observation / action specs contain a list of spaces,
          this is implicitly assumed across all Reaver components
    """

    def __init__(self, _id, render=False, reset_done=True, max_ep_len=None):
        self.id = _id
        self.render = render
        self.reset_done = reset_done
        self.max_ep_len = max_ep_len if max_ep_len else float('inf')

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def obs_spec(self):
        pass

    @abstractmethod
    def act_spec(self):
        pass


