import gym.spaces
import copy
import math, random
import gym.spaces
import numpy as np
import torch
from q_learning import utils
from q_learning.model import DQN
from torch.autograd import Variable

ALPHA = 0.95  # not using should be multiply with bellman error


class QLEARNING():
    def __init__(self, config, env):
        super(QLEARNING, self).__init__()

        self.env = env
        self.num_actions = env.action_space.n
        self.input_state_shape = env.observation_space.shape
        self.Q_function, self.target_Q_function = DQN(self.input_state_shape, self.num_actions), DQN(
            self.input_state_shape, self.num_actions)
        # self.Q_function = model.MyNetwork(config,(64,64,3),self.NUM_STATES)
        # self.target_Q_function = model.MyNetwork(config,(64,64,3),self.NUM_STATES)
        self.enable_gpu = config.enable_gpu
        if self.enable_gpu and torch.cuda.is_available():
            self.Q_function.cuda()
            self.target_Q_function.cuda()
        self.gamma = config.gamma
        self.epsilon = config.eps_start
        self.minimum_epsilon = config.eps_end
        self.epsilon_step = (config.eps_start - config.eps_end) / config.exploration_steps
        self.learn_step_counter = 0
        self.total_steps = 0

        self.memory = utils.ReplayMemory(config.MAX_REPLAY_MEMORY)
        self.minimum_memory = config.INITIAL_REPLAY_SIZE
        self.optimizer = torch.optim.Adam(self.Q_function.parameters(), lr=config.learning_rate)
        self.loss_func = utils.loss_criterion(config)
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        self.use_double_qlearning = config.use_ddqn

        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

    def update_target_model(self):
        self.target_Q_function.load_state_dict(self.Q_function.state_dict())

    def choose_action(self, state):
        if self.enable_gpu and torch.cuda.is_available():
            state = torch.unsqueeze(torch.FloatTensor(state).cuda(), 0)
        else:
            state = torch.unsqueeze(torch.FloatTensor(state), 0)

        # Decrease epsilon value
        # self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
        #                math.exp(-1. * self.step / EPSILON_DECAY)

        # do random action only till initial replay size fills
        if self.epsilon >= random.random() or self.memory.size() < self.minimum_memory:
            action = np.random.randint(0, self.num_actions)
        else:
            action_values = self.Q_function.forward(state)
            action = torch.max(action_values, 1)[1].data.cpu().numpy()[0]
        self.total_steps += 1

        # start decay after initial replay size fills
        if self.epsilon > self.minimum_epsilon and self.memory.size() >= self.minimum_memory:
            self.epsilon -= self.epsilon_step

        return action

    def store_transition(self, state, action, reward, next_state, is_terminated):
        done = 0
        if is_terminated:
            done = 1
        self.memory.push(state, action, reward, next_state, done)

    def learn(self, writer):

        if self.memory.size() < self.batch_size or self.memory.size() < self.minimum_memory:
            return

        # update the parameters
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_model()
            print("Target Q-function updated!!")

        # sample batch from memory
        batch_state, batch_action, batch_reward, batch_next_state, done_mask = self.memory.sample(self.batch_size)

        # print(batch_action)
        batch_state = Variable(torch.FloatTensor(np.float32(batch_state)))
        batch_next_state = Variable(torch.FloatTensor(np.float32(batch_next_state)))
        batch_action = Variable(torch.LongTensor(batch_action))
        batch_reward = Variable(torch.FloatTensor(batch_reward))
        done_mask = Variable(torch.FloatTensor(done_mask))

        # if config.CUSTOM_REWARD_SHAPING:
        #     batch_reward = np.clip(batch_reward, -1.0, +1.0)
        # high reward value can make training unstable. Thus Clipping Rewards technique clips scores,
        # which all positive rewards are set +1 and all negative rewards are set -1.

        # print(batch_state.shape)
        # print(batch_action)
        # print(batch_next_state.shape)
        # print(batch_reward.shape)
        # q_eval
        if self.enable_gpu and torch.cuda.is_available():
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_state = batch_state.cuda()
            batch_next_state = batch_next_state.cuda()
            done_mask = done_mask.cuda()

        # Compute current Q value, q_func takes only state and output value for every state-action pair
        # We choose Q based on action taken.
        current_Q_values = self.Q_function(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)

        # Compute next Q value based on which action gives max Q values from target network
        # next_Q_values = self.target_Q_function(batch_next_state).detach().max(1)[0]

        q_next = self.target_Q_function(batch_next_state).detach()
        if self.use_double_qlearning:
            best_actions = torch.argmax(self.Q_function(batch_next_state), dim=-1)
            next_Q_values = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        else:
            next_Q_values = q_next.max(1)[0]

        target_Q_values = batch_reward + self.gamma * next_Q_values * (1 - done_mask)

        # loss = (current_Q_values - Variable(target_Q_values.data)).pow(2).mean()
        # loss = (target_Q_values - current_Q_values).pow(2).mul(0.5).mean()

        loss = self.loss_func(current_Q_values, target_Q_values)
        writer.add_scalar("loss Per batch", loss, global_step=self.total_steps)
        writer.add_scalar("epsilon per batch", self.epsilon, global_step=self.total_steps)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1


def main():
    print("hello world")


if __name__ == '__main__':
    main()
