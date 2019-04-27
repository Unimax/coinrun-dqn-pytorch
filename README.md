Coinrun Base code is taken from https://github.com/openai/coinrun

![CoinRun](coinrun.png?raw=true "CoinRun")

Setup coinrun:-

# Linux with Python 3.6
```
apt-get install mpich build-essential qt5-default pkg-config
```
# Mac with Python 3.6
```
brew install qt open-mpi pkg-config
```

```
git clone https://github.com/unimax/coinrun-dqn-pytorch.git
cd coinrun-dqn-pytorch/Project-Code/
pip install tensorflow==1.12.0 tensorflow-gpu==1.12.0
pip install -r requirements.txt
pip install -e .
```

## Try it out

Try the environment out with the keyboard:

```
python -m coinrun.interactive
```

If this fails, you may be missing a dependency or may need to fix `coinrun/Makefile` for your machine.
For tenserflow based PPO traning check out commands from orignal coinrun repo https://github.com/openai/coinrun

--------------------------------------------------------------------------------------------------------------

Start DQN Learning:-

```
python -m q_learning.solution --num-levels 1 --set-seed 1
```

for more custimization option and traning parameters check Project-code/q_learning/config.py

output graphs in logs dir (orange is when --CUSTOM_REWARD_SHAPING True):-

![ep_length](ep_length.png?raw=true "ep_length")

![ep_reward](ep_reward.png?raw=true "ep_reward")
Blue default reward settings

Following graphs clearly show the difference in perfomance in DQN and Double DQN :-

![dqnVSddqnEpReward](dqnVSddqnEpReward.png?raw=true "dqnVSddqnEpReward")

![dqnVsDdqnEpLength](dqnVsDdqnEpLength.png?raw=true "dqnVsDdqnEpLength")

![dqnVSddqnAvgRew](dqnVSddqnAvgRew.png?raw=true "dqnVSddqnAvgRew")

![ddqnLegend](ddqnLegend.png?raw=true "ddqnLegend")




## Colab

There's also a [Colab notebook](https://colab.research.google.com/drive/1e2Eyl8HANzcqPheVBMbdwi3wqDv41kZt) showing how to setup CoinRun.

See [LICENSES](ASSET_LICENSES.md) for asset license information.
