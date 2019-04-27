import os
import numpy as np
import torch, copy
import time, datetime
import q_learning.config as dqnconfig
from q_learning.model import MyNetwork, DQN
from tensorboardX import SummaryWriter
from q_learning import utils
from q_learning.QLearning import QLEARNING
from coinrun import make
import coinrun.main_utils as coinrun_utils
from coinrun import setup_utils
from coinrun.config import Config as conrun_config


def rewardShaping(action, already_done, done, ep_length, reward, state, next_state):
    if reward > 0:
        reward = reward * 100
        already_done = True
    elif already_done:
        print("ep ended with success")
    elif done and reward == 0 and ep_length > 999:
        print("ep ended with timeout")
        reward = -1
    elif done and reward == 0:
        print("ep ended with death")
        reward = -500
    elif np.array_equal(state, next_state):
        reward = -2
    elif action == 4 or action == 1:  # going right is the correct direction
        reward = 1
    else:
        reward = -1
    return reward, already_done


def train(config):
    """Training process of DQN.

    """

    # Initialize the envrioment
    env = utils.Scalarize(coinrun_utils.make_general_env(1, seed=1))

    # Create log directory and save directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Create summary writer
    start_time = time.time()
    st = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

    tr_writer = SummaryWriter(log_dir=os.path.join(config.log_dir,
                                                   "DQN Traning {} c_rew {} numlvl {} seed {}".format(st,
                                                                                                      config.CUSTOM_REWARD_SHAPING,
                                                                                                      conrun_config.NUM_LEVELS,
                                                                                                      conrun_config.SET_SEED)))

    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(config.save_dir, "checkpoint.pth")

    # Initialize training
    dqn = QLEARNING(config, env)

    # Make sure that the model is set for training
    dqn.Q_function.train()
    dqn.target_Q_function.eval()  # target is never learning but getting copied from Q_function

    # Check for existing training results. If it existst, and the configuration
    # is set to resume `config.resume==True`, resume from previous training. If
    # not, delete existing checkpoint.
    if os.path.exists(checkpoint_file):
        if config.resume:
            print("Checkpoint found! Resuming")
            # Read checkpoint file.
            load_res = torch.load(
                checkpoint_file,
                map_location="cpu")
            dqn.Q_function.load_state_dict(load_res["model"])
            dqn.update_target_model()
            dqn.optimizer.load_state_dict(load_res["optimizer"])
        else:
            os.remove(checkpoint_file)
    max_avg_reward = 0.04
    # Training loop
    for i in range(config.num_episodes):
        state = env.reset()
        ep_reward = 0
        ep_length = 0
        terminated = False
        while True:
            if config.render_play:
                env.render()
            action = dqn.choose_action(state)

            next_state, reward, done, info = env.step(action)

            if config.CUSTOM_REWARD_SHAPING:
                reward, terminated = rewardShaping(action, terminated, done, ep_length, reward, state, next_state)
            ep_length += 1
            ep_reward += reward

            dqn.store_transition(state, action, reward, next_state, done)

            dqn.learn(tr_writer)

            if done:
                break
            state = copy.copy(next_state)
        print("finished ep: {} , ep_rew {} len {}. esp {}. time_passed {}, memory_size {} ".format(i,
                                                                                                   ep_reward,
                                                                                                   ep_length,
                                                                                                   dqn.epsilon,
                                                                                                   time.time() - start_time,
                                                                                                   dqn.memory.size()))
        tr_writer.add_scalar("ep_length", ep_length, global_step=i)
        ep_reward_norm = ep_reward
        if config.CUSTOM_REWARD_SHAPING:
            ep_reward_norm = ep_reward / 100  # so it will be in the same range as default reward in graph ploting
        avg_reward_norm = ep_reward_norm / ep_length

        tr_writer.add_scalar("ep_reward", ep_reward_norm, global_step=i)
        tr_writer.add_scalar("Avg. reward per step", avg_reward_norm, global_step=i)
        if config.test_while_train and avg_reward_norm > max_avg_reward and dqn.memory.size() > dqn.minimum_memory:
            if test(config, dqn.Q_function, 3):
                torch.save({
                    "current_ep": i,
                    "model": dqn.Q_function.state_dict(),
                    "optimizer": dqn.optimizer.state_dict(),
                }, os.path.join(config.save_dir,
                                "bestmodel_{}_cRew_{}_numlvl_{}_seed_{}.pth".format(i,
                                                                                    config.CUSTOM_REWARD_SHAPING,
                                                                                    conrun_config.NUM_LEVELS,
                                                                                    conrun_config.SET_SEED)))
                max_avg_reward = avg_reward_norm
            dqn.Q_function.train()

        if (i % 50) == 0:  # hardcoded save checkpoint interval
            torch.save({
                "current_ep": i,
                "model": dqn.Q_function.state_dict(),
                "optimizer": dqn.optimizer.state_dict(),
            }, checkpoint_file)

    torch.save({
        "current_ep": i,
        "model": dqn.Q_function.state_dict(),
        "optimizer": dqn.optimizer.state_dict(),
    }, os.path.join(config.save_dir,
                    "final_{}_rShaping_{}_numlvl_{}_seed_{}_time_{}.pth".format(i, config.CUSTOM_REWARD_SHAPING,
                                                                                conrun_config.NUM_LEVELS,
                                                                                conrun_config.SET_SEED, st)))
    env.close()


def test(config, agent=None, levels=5):
    """Test routine"""

    env = utils.Scalarize(make('standard', num_envs=1))

    if agent is None:
        print("Testing numlvl {} seed {} file: {}".format(conrun_config.NUM_LEVELS, conrun_config.SET_SEED,
                                                          config.model_filename))
        agent = DQN(env.observation_space.shape, env.action_space.n)
        if config.enable_gpu and torch.cuda.is_available():
            agent = agent.cuda()
        bestmodel_file = os.path.join(config.save_dir, config.model_filename)
        load_res = torch.load(bestmodel_file, map_location="cpu")
        agent.load_state_dict(load_res["model"])
    else:
        config.render_play = False

    agent.eval()
    success = 0
    total_steps = 0
    for i in range(levels):
        state = env.reset()
        ep_reward = 0
        ep_length = 0
        while True:
            if config.render_play:
                env.render()
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            action = torch.max(agent.forward(state), 1)[1].data.numpy()[0]  # TODO debug this

            next_state, reward, done, info = env.step(action)

            ep_length += 1
            ep_reward += reward

            state = copy.copy(next_state)

            if done:
                print("test episode: {} , the episode reward : {} with length : {}".format(i, ep_reward, ep_length))
                break

        if ep_reward > 0:
            success = success + 1
        total_steps += ep_length

    print("Testing result : {} % completed. Avg. ep length : {}".format(success / levels * 100, total_steps / levels))
    env.close()
    if success >= (levels / 2):
        return True
    return False


def main():
    """The main function."""

    setup_utils.setup_and_load(is_high_res=True)

    config, unparsed = dqnconfig.get_config()
    # ----------------------------------------
    # Parse configuration
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print("unparsed for DQN :", unparsed)
        # input("Press Enter to continue...")

    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":
    main()

