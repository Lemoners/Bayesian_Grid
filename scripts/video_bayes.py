import os
import sys
import json
import numpy as np
from mygrid.MiniGrid.AE.VAE import VAE
from mygrid.MiniGrid.Generator.HyperPara import GRID_WIDTH, GRID_HEIGHT
from mygrid.MiniGrid.Utils.BFS import BFSAgent
from mygrid.MiniGrid.Generator.HardMazeGene import HardMazeGene
from mygrid.MiniGrid.Generator import MazeGene
from mygrid.MiniGrid.Generator.HyperPara import Z_DIM
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib
import random
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from gym_minigrid.envs.maze import RenderMazeEnv9x9
import argparse
cur_dir = os.path.dirname(os.path.abspath((__file__)))
para_dir = os.path.join(cur_dir, "para")
msavefig = para_dir + "/vis"
if not os.path.exists(msavefig):
    os.makedirs(msavefig)

parser = argparse.ArgumentParser(description="Video bayes maze")
parser.add_argument("-i", "--interval", type=int, default=2)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE().to(device)
    vae.eval()

    with open(para_dir + "/historyBayesGridEnv.json", "r+") as f:
        history = json.load(f)

    args = parser.parse_args()
    # print(args)

    fig, ax = plt.subplots()
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    len_his = len(history[::args.interval])
    print("Process {} img".format(len_his))
    for i, h in enumerate(history[::args.interval]):
        print("\r", i, end="", flush=True)
        env = RenderMazeEnv9x9(_grid=np.array(h))
        img = env.render(mode="non-human")
        imshow_obj = ax.imshow(img)
        imshow_obj.set_data(img)
        fig.canvas.draw()
        plt.savefig(msavefig + "/{:0>10}.png".format(i), dpi=300)







