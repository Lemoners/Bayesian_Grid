import mygrid
import gym
from mygrid.MiniGrid.Generator import MazeGene
# env = gym.make('basicgrid-v0')
# env = gym.make('validgrid-v0')
# print(env.reset())

maze = MazeGene()
print(maze.gene()[0])
