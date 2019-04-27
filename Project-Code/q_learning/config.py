

import argparse

# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for the main program
main_arg = add_argument_group("Main")

main_arg.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "test"],
                      help="Run mode : test or train?")



# ----------------------------------------
# Arguments for DQN training
train_arg = add_argument_group("Training")

train_arg.add_argument("--model_filename", type=str,  # currently not working. rename as checkpoint if you wanna test
                      default="checkpoint.pth",
                      help="exact filename to grab for testing. relative to save_dir")

train_arg.add_argument("--use_ddqn", type=str2bool,
                       default=False,
                       help="force use Double DQN")

train_arg.add_argument("--test_while_train", type=str2bool,
                       default=False,
                       help="perform testing while traning and save the best model")

train_arg.add_argument("--enable_gpu", type=str2bool,
                       default=True,
                       help="force disable gpu")

train_arg.add_argument("--render_play", type=str2bool,
                       default=True,
                       help="render the envrioment")

train_arg.add_argument("--learning_rate", type=float,
                       default=0.00025,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--num_episodes", type=int,
                       default=500,
                       help="Number of episodes to train for")

train_arg.add_argument("--batch_size", type=int,
                       default=64,
                       help="Q-learning training batch size")

train_arg.add_argument("--eps_start", type=float,
                       default=1.0,
                       help="e-greedy threshold start value")

train_arg.add_argument("--eps_end", type=float,
                       default=0.1,
                       help="e-greedy threshold end value")

train_arg.add_argument("--exploration_steps", type=int,
                       default=20000,
                       help="Steps over which the initial value of epsilon is linearly annealed to its esp_end")

train_arg.add_argument("--gamma", type=float,
                       default=0.99,
                       help="Discount factor gamma")

train_arg.add_argument("--INITIAL_REPLAY_SIZE", type=int,
                       default=10000,
                       help="Number of steps to populate the replay memory before training starts")

train_arg.add_argument("--MAX_REPLAY_MEMORY", type=int,
                       default=60000,
                       help="Number of replay memory the agent uses for training")

train_arg.add_argument("--LEARNING_FREQ", type=int,  # --- NOT USING
                       default=4,
                       help="How many steps to take between learn activity")

train_arg.add_argument("--target_update_freq", type=int,
                       default=1000,
                       help="update target Q-function after how many Q-function updates")


train_arg.add_argument("--CUSTOM_REWARD_SHAPING", type=str2bool,
                       default=True,
                       help="Whether to resume training from existing checkpoint")

train_arg.add_argument("--val_intv", type=int,
                       default=1000,
                       help="Validation interval")

train_arg.add_argument("--rep_intv", type=int,
                       default=1000,
                       help="Report interval")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs",
                       help="Directory to save old_logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save the best model")

train_arg.add_argument("--resume", type=str2bool,
                       default=False,
                       help="Whether to resume training from existing checkpoint")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
