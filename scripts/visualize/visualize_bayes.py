import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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



para_save = os.path.dirname(os.path.abspath(__file__)) + "/para/historyBayesGridEnv.json"
msavefig = os.path.dirname(os.path.abspath(__file__)) + "/para/pic"
if not os.path.exists(msavefig):
    os.makedirs(msavefig) 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open(para_save, "r+") as f:
    history = json.load(f)
vae = VAE().to(device).eval()
bfs = BFSAgent()


bayes_history = []
print("")
len_his = len(history)
for i, h in enumerate(history):
    print("\rGenerating {:.4f}% HardMazeGrid".format(100*i/len_his))
    grid = vae.decode(torch.from_numpy(np.array(h)).float().unsqueeze(0).to(device))
    grid = grid.cpu().round().detach().numpy().reshape(-1)
    bayes_history.append(grid)

print("Processing {} grids".format(len(bayes_history)))

samples = len(history)

hardgene = HardMazeGene()
hardmaze_history = []
for _ in range(samples):
    maze, _, _ = hardgene.gene()
    maze[np.where(maze==2)] = 0
    maze[np.where(maze==3)] = 0
    hardmaze_history.append(maze.reshape(-1))


print("Start dimension reduction")
pca = PCA(n_components=2)
bayes_history = pca.fit_transform(bayes_history)
hardmaze_history = pca.fit_transform(hardmaze_history)

tsne = TSNE(n_components=2)
# bayes_history = tsne.fit_transform(bayes_history)
# hardmaze_history = tsne.fit_transform(hardmaze_history)
print("End dimension reduction")

bayes_history = normalize(bayes_history, norm="l2" , axis=0)
hardmaze_history = normalize(hardmaze_history, norm="l2", axis=0)

bayes_history = list(zip(*list(bayes_history)))
hardmaze_history = list(zip(*list(hardmaze_history)))

colors1 = '#00CED1' #点的颜色
colors2 = '#DC143C'
area = np.pi * 4**2  # 点面积 
# 画散点图
print("Start Drawing")

alp = [0.05 + 0.9 * i / samples for i in range(samples)]
plt.scatter(*bayes_history, s=area, c=alp, cmap=plt.cm.Blues, label='BAYES')
plt.scatter(*hardmaze_history, s=area, c=colors2, alpha=0.8, label='HARD_MAZE')

plt.legend()

mtime = datetime.datetime.now()
mtime = datetime.datetime.strftime(mtime,'%m_%d_%H_%M_%S')


print("SAVE in", msavefig + "/" + mtime + ".jpg")
plt.savefig(msavefig + "/" + mtime + ".jpg", dpi=1000)
plt.show()





