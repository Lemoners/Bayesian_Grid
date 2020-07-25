import mygrid
import gym
from mygrid.MiniGrid.Generator import MazeGene
from mygrid.MiniGrid import RenderWrapper
import random
import time


# env = gym.make('basicgrid-v0')
env = gym.make('validgrid-v0')
env = RenderWrapper(env)
print(env.reset())
while True:
    action = eval(input())
    print(action)
    _, _, done, _ = env.step(action)
    # time.sleep(2)
    if done:
        break

# maze = MazeGene()
# print(maze.gene()[0])
