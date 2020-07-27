import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mygrid
import gym
from mygrid.MiniGrid.Generator import MazeGene
from mygrid.MiniGrid import RenderWrapper
import random
import time
import numpy as np
from mygrid.MiniGrid.Utils.BFS import BFSAgent
from mygrid.MiniGrid.Discriminator.HardMazeDiscriminator import HardMazeDiscriminator
from mygrid.MiniGrid.Generator.HardMazeGene import HardMazeGene
import torch
from mygrid.MiniGrid.Agent import ILNet
from mygrid.MiniGrid.Discriminator.AgentDiscriminator import AgentDiscriminator


# env = gym.make('basicgrid-v0')
env = gym.make('mazegrid-v0')
# env = RenderWrapper(env)
# model = BFSAgent()

# for _ in range(100):
#     obs = env.reset()
#     while True:
#         h, s = model.solve(obs)
#         action = h[0][1]
#         obs, _, done, _ = env.step(action)
#         time.sleep(0.1)
#         if done:
#             break
model = ILNet()
checkpoint = torch.load(os.path.dirname(os.path.abspath(__file__)) + "/model/IL/Basic/MazeGridEnv")
model.load_state_dict(checkpoint['5000'])

# for i in range(1):
#     # grid = env.reset()
#     grid = np.array([[2, 1, 0, 0, 0],\
#                      [0, 1, 0, 0, 0],\
#                      [0, 1, 0, 0, 0],\
#                      [0, 1, 1, 1, 1],\
#                      [0, 0, 0 ,0 ,3]])
#     grid = np.array([[2, 0, 0, 0, 0],\
#                      [1, 0, 1, 1, 0],\
#                      [0, 0, 0, 0, 0],\
#                      [0, 1, 0, 1, 1],\
#                      [0, 0, 0 ,0 ,3]])
#     discriminator = HardMazeDiscriminator()
#     print(discriminator.evaluate_maze(grid))

gene = HardMazeGene()
grids = gene.batch_gene(batches=10)

discriminator = AgentDiscriminator()
print(discriminator.evaluate_agent(model, grids))


