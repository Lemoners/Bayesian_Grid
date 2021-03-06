"""
Evaluate RL agents on MazeGridEnv, ValidGridEnv, HardGridEnv
"""
from mygrid.MiniGrid import MazeGridEnv, ValidGridEnv, RenderWrapper
from mygrid.MiniGrid.HardGridEnv import HardGridEnv

import os
import sys
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

RL_EVALUATE_ITER_EVERY_TEST = 100
RL_EVALUATE_EPOCH_EVERY_ITER = 5

def smooth(data, sm=1):
    if sm > 1:
        y = np.ones(sm)*1.0/sm
        data = np.convolve(y, data, "same")
    return data

model_dir = os.path.dirname(os.path.abspath(__file__)) + '/data/model/RL/A2C'
msavefig = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'RL')
if not os.path.exists(msavefig):
    os.makedirs(msavefig)

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sparse', default=False, action='store_true')


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.sparse:
        model = A2C.load(model_dir + "/MazeGridEnv/A2C.zip")
    else:
        model = A2C.load(model_dir + "/SparseMazeGridEnv/A2C.zip")

    envs = {"MazeGridEnv": MazeGridEnv, "ValidGridEnv": ValidGridEnv, "HardGridEnv": HardGridEnv}

    for k in envs.keys():
        print("\nEvaluate RL agent on {} Env".format(k))
        test_history = []
        for i in range(3):
            result = []
            env = envs[k]()
            # env = RenderWrapper(env)
            for p in range(RL_EVALUATE_ITER_EVERY_TEST):
                finish = []
                print("\r","%.2f" % (100*i/3 + 100*p/RL_EVALUATE_ITER_EVERY_TEST/3), \
                    '*' * int(10*i/3 + 10*p/RL_EVALUATE_ITER_EVERY_TEST/3), end="",flush=True)
                for _ in range(RL_EVALUATE_EPOCH_EVERY_ITER):
                    obs = env.reset()
                    while True:
                        action, _states = model.predict(obs, deterministic=True)
                        obs, reward, done, info = env.step(action)
                        if done:
                            if info.get("success") == True:
                                finish.append(1)
                            else:
                                finish.append(0)
                            break
                result.append(np.mean(finish))
            test_history.append(result)
        i_data = []
        for line in test_history:
            line = smooth(line, 6)
            for index, result in enumerate(line):
                i_data.append([index, result])
        i_data = pd.DataFrame(i_data, columns=["episode", "reward"])
        sns.lineplot(x="episode", y="reward",data=i_data)
        plt.title("RL Test in {} Env".format(k), fontsize=20)
        plt.savefig(msavefig + "/" + k +".jpg")


