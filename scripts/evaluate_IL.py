import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mygrid.MiniGrid.BayesGridEnv import BayesGridEnv, RandomBayesGridEnv
from mygrid.MiniGrid import BasicGridEnv, ValidGridEnv, MazeGridEnv, RenderWrapper
from mygrid.MiniGrid.HardGridEnv import HardGridEnv
from mygrid.MiniGrid.Agent import ILNet
from mygrid.MiniGrid.Utils import smooth
import argparse
import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--models', nargs='+', default=['RandomBayesGridEnv','BayesGridEnv', 'BasicGridEnv', 'ValidGridEnv', 'MazeGridEnv'])
parser.add_argument('-e', '--envs', nargs='+', default=['HardGridEnv', 'ValidGridEnv', 'MazeGridEnv', 'RandomBayesGridEnv'])
parser.add_argument('-v', '--visual', action="store_true", default=False)
parser.add_argument('-n', '--num', default=10)
parser.add_argument('--log', default="")
parser.add_argument('--length', type=int, default=50)
parser.add_argument('--save', action="store_true", default=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(env, model, model_str, model_episode, episodes=30, sm=5):
    model = model.to(device)
    model.eval()

    data = []
    history = []
    for _ in range(3):
        for i in range(episodes):
            obs = env.reset()
            while True:
                action = model.predict(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    if info.get('success'):
                        history.append(1)
                    else:
                        history.append(0)
                    break
        history = sorted(smooth(history, sm=sm))
        data.append([model_episode, model_str, np.mean(history)])

    
    # data.append([model_episode, model_str, np.quantile(history, 0.3)])
    # data.append([model_episode, model_str, np.quantile(history, 0.5)])
    # data.append([model_episode, model_str, np.quantile(history, 0.7)])

    # for r in history:
    #     data.append([model_episode, model_str, r])
    return data

if __name__ == "__main__":
    model_checkpoint_path = os.path.dirname(os.path.abspath(__file__)) + "/model/IL/Basic"
    args = parser.parse_args()

    pic_dir = os.path.dirname(os.path.abspath(__file__)) + "/results"
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    mtime = datetime.datetime.now()
    mtime = datetime.datetime.strftime(mtime, "%m_%d_%H_%M_%S")

    if args.visual:
        env = RenderWrapper(eval(args.envs[0])())
        model_checkpoint = torch.load(model_checkpoint_path + "/" + args.models[0] + args.log)
        k = list(model_checkpoint.keys())[-1]
        # print(len(list(model_checkpoint.keys())))
        # k = list(model_checkpoint.keys())[200]
        model = ILNet()
        model.load_state_dict(model_checkpoint[k])
        model = model.to(device)
        model.eval()
        for _ in range(args.num):
            obs = env.reset()
            while True:
                action = model.predict(obs)
                obs, reward, done, info = env.step(action)
                time.sleep(0.1)
                if done:
                    break
    else:
        for _env in args.envs:
            history = []
            plt.clf()
            for _model in args.models:
                env = eval(_env)()
                model_checkpoint = torch.load(model_checkpoint_path + "/" + _model + args.log, map_location=device)
                # print(list(model_checkpoint.keys()))
                for i, k in enumerate(list(model_checkpoint.keys())[:args.length]):
                    print("\rEvaluating {} on {} {:.2f}%".format(_model, _env, 100*(i/len(list(model_checkpoint.keys())[:args.length]))), end="", flush=True)
                    model = ILNet()
                    model.load_state_dict(model_checkpoint[k])
                    index = eval(k)
                    # if "Bayes" in _model:
                        # index *= 4
                    data = evaluate(env, model, _model, index)
                    history.extend(data)
                print("")
            history = pd.DataFrame(history, columns=['episode', "model", "reward"])
            # print(history)

            for _model in args.models:
                rs = list(history.loc[history["model"]==_model, "reward"])
                rs = smooth(rs, sm=5)
                history.loc[history["model"]==_model, "reward"] = rs
            # print(history)
            # sns.set(style="darkgrid")
            sns.lineplot(x="episode", y="reward", hue='model', data=history)
            plt.title("IL Agent Test in {} Env".format(_env), fontsize=20)
            plt.xlabel("Update Iter")
            if args.save:
                plt.savefig(pic_dir + "/" + _env + mtime + ".jpg", dpi=300)
            else:
                plt.show()




