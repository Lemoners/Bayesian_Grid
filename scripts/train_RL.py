import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from mygrid.MiniGrid import MazeGridEnv, SparseRewardWrapper

SPARSE = True

model_save = os.path.dirname(os.path.abspath(__file__)) + "/model/RL/PPO"

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = MazeGridEnv()
        if SPARSE:
            env = SparseRewardWrapper(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    time_steps = 50000
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=time_steps)

    if not SPARSE:
        model_save += "/MazeGridEnv"
    else:
        model_save += "/SparseMazeGridEnv"

    if not os.path.exists(model_save):
        os.makedirs(model_save)

    model.save(model_save + "/{}".format(time_steps))