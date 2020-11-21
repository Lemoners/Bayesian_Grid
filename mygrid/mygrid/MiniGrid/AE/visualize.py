import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib
import torch
from .VAE import VAE
from ..Generator import MazeGene
from ..Generator.HyperPara import Z_DIM, GRID_HEIGHT, GRID_WIDTH
from ..Utils.BFS import BFSAgent
import random
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize(samples=5, msavefig=""):
    vae = VAE().to(device)
    vae.eval()
    
    vae_history = []
    for _ in range(samples):
        with torch.no_grad():
            z = torch.randn(Z_DIM).to(device)
            out = vae.decode(z).view(-1).cpu().numpy()
            vae_history.append(out.copy())
    
    gene = MazeGene()
    maze_history = []
    for _ in range(samples):
        maze, _, _ = gene.gene()
        maze[np.where(maze==2)] = 0
        maze[np.where(maze==3)] = 0
        maze_history.append(maze.reshape(-1))
    
    pca = PCA(n_components=2)
    # vae_history = pca.fit_transform(vae_history)
    # maze_history = pca.fit_transform(maze_history)

    tsne = TSNE(n_components=2)
    vae_history = tsne.fit_transform(vae_history)
    maze_history = tsne.fit_transform(maze_history)


    vae_history = normalize(vae_history, norm="l2" , axis=0)
    maze_history = normalize(maze_history, norm="l2", axis=0)

    vae_history = zip(*list(vae_history))
    maze_history = zip(*list(maze_history))

    colors1 = '#00CED1' #点的颜色
    colors2 = '#DC143C'
    area = np.pi * 4**2  # 点面积 
    # 画散点图
    plt.scatter(*vae_history, s=area, c=colors1, alpha=0.4, label='VAE')
    plt.scatter(*maze_history, s=area, c=colors2, alpha=0.4, label='MAZE')

    plt.legend()
    
    print("SAVE in", msavefig)
    plt.savefig(msavefig, dpi=300)
    plt.show()

def test_solve(n=1000):
    vae = VAE().to(device)
    vae.eval()

    bfs = BFSAgent()

    solvable = 0

    for _ in range(n):
        with torch.no_grad():
            z = torch.randn(Z_DIM).to(device)
            # print(z)
            out = vae.decode(z).view(GRID_HEIGHT, GRID_WIDTH).round().int().cpu().numpy()
            out[0][0] = 2
            out[GRID_HEIGHT-1][GRID_WIDTH-1] = 3
            # print(out)
            # input()
            _, s = bfs.solve(out)
            if s:
                solvable += 1
    return solvable / n
            
    

