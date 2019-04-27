import numpy as np
from q_learning.utils import Scalarize
from coinrun import make
from coinrun import setup_utils



def testing():
    setup_utils.setup_and_load()
    episodes = 10
    env = Scalarize(make('standard', num_envs=1))
    for i in range(episodes):
        env.reset()
        while True:
            env.render()
            action = np.random.randint(0, env.action_space.n)
            next_state, reward, done, info = env.step(action)
            if done or reward > 0:
                break


def main():
    testing()


if __name__ == '__main__':
    main()
