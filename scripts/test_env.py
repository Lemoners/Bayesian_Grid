import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mygrid.MiniGrid import MazeGridEnv, SparseRewardWrapper, RenderWrapper


def get_action(key):
    if key == "a":
        return 3
    if key == "s":
        return 2
    if key == "w":
        return 1
    if key == "d":
        return 4
    return 0

env = MazeGridEnv()
env = RenderWrapper(env)
for i in range(10):
    obs = env.reset()
    while True:
        action = get_action(input(""))
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            break
