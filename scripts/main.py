import mygrid
import gym
from mygrid.MiniGrid.Generator import MazeGene
from mygrid.MiniGrid import RenderWrapper
import random
import time
from mygrid.MiniGrid.Utils.BFS import BFSAgent


# env = gym.make('basicgrid-v0')
env = gym.make('mazegrid-v0')
env = RenderWrapper(env)
model = BFSAgent()

for _ in range(100):
    obs = env.reset()
    while True:
        h, s = model.solve(obs)
        action = h[0][1]
        obs, _, done, _ = env.step(action)
        time.sleep(0.1)
        if done:
            break
