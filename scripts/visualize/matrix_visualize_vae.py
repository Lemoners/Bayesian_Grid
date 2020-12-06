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
msavefig = para_dir + "/matrix"
if not os.path.exists(msavefig):
    os.makedirs(msavefig)

parser = argparse.ArgumentParser(description="Video bayes maze")
parser.add_argument("-i", "--interval", type=int, default=2)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE().to(device)
    vae.eval()

    
    # ax.set_xticks([], [])
    # ax.set_yticks([], [])

    dim_len = 5
    delta = float(10 / dim_len)
    fig, ax = plt.subplots(nrows=dim_len, ncols=dim_len)
    start = np.random.randn(12).astype(np.float64)
    axes = ax.flatten()
    for i in range(dim_len):
        for j in range(dim_len):
            # plt.subplot(dim_len*dim_len, j+1, i+1)
            # plt.subplot(dim_len*dim_len, j+1, i+1)
            print("\rProcess {:.1f}%".format(100*(i*dim_len+j)/dim_len/dim_len), end="", flush=True)
            h = start.copy()
            h[0] += delta * i
            h[1] += delta * j
            grid = vae.decode(torch.from_numpy(np.array(h)).float().unsqueeze(0).to(device))
            grid = grid.cpu().round().detach().numpy().reshape(GRID_HEIGHT, GRID_WIDTH).astype(np.uint8)
            grid[0, 0] = 2
            grid[GRID_HEIGHT-1, GRID_WIDTH-1] = 3
            env = RenderMazeEnv9x9(_grid=grid)
            img = env.render(mode="non-human")
            axes[i*dim_len+j].set_xticks([], [])
            axes[i*dim_len+j].set_yticks([], [])
            imshow_obj = axes[i*dim_len+j].imshow(img)
            imshow_obj.set_data(img)
            if (i == 0):
                axes[i*dim_len+j].set_title("{:.2f}".format(h[1]))
            if (j == 0):
                axes[i*dim_len+j].set_ylabel("{:.2f}".format(h[0]))
                # set_title("{:.1f},{:.1f},{:.1f},{:.1f}".format(*h))
            # fig.canvas.draw()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    # plt.tight_layout()
    plt.show()







