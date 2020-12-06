"""
Train RL agents using PPO with stable_baseline3 (https://github.com/DLR-RM/stable-baselines3)
"""

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import argparse
import os
from mygrid.MiniGrid import MazeGridEnv, SparseRewardWrapper, RenderWrapper
from mygrid.MiniGrid.Utils import conv2d_size_out
from mygrid.MiniGrid.Generator.HyperPara import GRID_HEIGHT, GRID_WIDTH

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sparse', default=False, action='store_true')

model_save = os.path.dirname(os.path.abspath(__file__)) + "/data/model/RL/A2C"
if not os.path.exists(model_save):
    os.makedirs(model_save)

def make_env(rank, sparse=False, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = MazeGridEnv()
        if sparse:
            env = SparseRewardWrapper(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    args = parser.parse_args()

    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i, sparse=args.sparse) for i in range(num_cpu)])

    time_steps = 200000
    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=time_steps)

    if not args.sparse:
        model_save += "/MazeGridEnv"
    else:
        model_save += "/SparseMazeGridEnv"

    if not os.path.exists(model_save):
        os.makedirs(model_save)

    model.save(model_save + "/model_{}".format(time_steps))