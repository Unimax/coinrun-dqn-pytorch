import random
import numpy as np
import gym
import torch
from collections import namedtuple, deque
from torch import nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


def conv2d_size_out(size, kernel_size, stride):
    return (size - kernel_size) // stride + 1


def loss_criterion(config):
    """Returns the loss object based on the commandline argument for the data term

    """

    if config.loss_type == "cross_entropy":
        data_loss = nn.CrossEntropyLoss()
    elif config.loss_type == "svm":
        data_loss = nn.MultiMarginLoss()
    else:
        data_loss = nn.MSELoss()

    return data_loss


def model_criterion(config):  # not using in DQN
    """Loss function based on the commandline argument for the regularizer term"""

    def model_loss(model):
        loss = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                loss += torch.sum(param ** 2)

        return loss * config.l2_reg

    return model_loss


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # if self.size() >= self.buffer.maxlen - 1:
        #     return #dont store new exp
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def size(self):
        return len(self.buffer)

    def __len__(self):
        return self.size()


class Scalarize:
    """
    Convert a VecEnv into an Env

    There is a minor difference between this and a normal Env, which is that
    the final observation (when done=True) does not exist, and instead you will
    receive the second to last observation a second time due to how the VecEnv
    interface handles resets.  In addition, you are cannot step this
    environment after done is True, since that is not possible for VecEnvs.
    """

    def __init__(self, venv) -> None:
        assert venv.num_envs == 1
        self._venv = venv
        self._waiting_for_reset = True
        self._previous_obs = None
        self.observation_space = self._venv.observation_space
        self.action_space = self._venv.action_space
        self.metadata = self._venv.metadata
        # self.spec = self._venv.spec
        self.reward_range = self._venv.reward_range

    def _process_obs(self, obs):
        if isinstance(obs, dict):
            # dict space
            scalar_obs = {}
            for k, v in obs.items():
                scalar_obs[k] = v[0]
            return scalar_obs
        else:
            return obs[0]

    def reset(self):
        self._waiting_for_reset = False
        obs = self._venv.reset()
        self._previous_obs = obs
        return self._process_obs(obs)

    def step(self, action):
        assert not self._waiting_for_reset
        final_action = action
        if isinstance(self.action_space, gym.spaces.Discrete):
            final_action = np.array([action], dtype=self._venv.action_space.dtype)
        else:
            final_action = np.expand_dims(action, axis=0)
        obs, rews, dones, infos = self._venv.step(final_action)
        if dones[0]:
            self._waiting_for_reset = True
            obs = self._previous_obs
        else:
            self._previous_obs = obs
        return self._process_obs(obs), rews[0], dones[0], infos[0]

    def render(self, mode="human"):
        if mode == "human":
            return self._venv.render(mode=mode)
        else:
            return self._venv.get_images(mode=mode)[0]

    def close(self):
        return self._venv.close()

    def seed(self, seed=None):
        return self._venv.seed(seed)

    @property
    def unwrapped(self):
        # it might make more sense to return the venv.unwrapped here
        # except that the interface is different for a venv so things are unlikely to work
        return self

    def __repr__(self):
        return f"<Scalarize venv={self._venv}>"
