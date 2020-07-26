import gym
import numpy as np
from gym import spaces
from .Utils import direction2action, action2direction, point_distance
from .Generator.HyperPara import *
from .Generator import BasicGene

ACTION_SPACE_DIM = 5


class BasicGridEnv(gym.Env):
    def __init__(self, generator=BasicGene()):
        super(BasicGridEnv, self).__init__()
        self.generator = generator

        # Reset
        self.grid, self.pos, self.goal_pos = self.generator.gene()
        self.h, self.w = self.grid.shape
        self.max_steps = self.h * self.w

        # need reset each time
        self.dis2goal = point_distance((0, 0), self.goal_pos)
        self.step_count = 0

        # action space && observation space
        self.action_space = spaces.Discrete(ACTION_SPACE_DIM)
        self.observation_space = spaces.Box(
            low=0, high=3, shape=self.grid.shape, dtype=np.uint8)

    def step(self, action):
        x, y = self.pos
        x += action2direction(action)[0]
        y += action2direction(action)[1]

        reward = 0
        done = False
        info = {}

        if (0 <= x < self.w and 0 <= y < self.h):
            if (self.grid[y][x] == GOAL):
                reward += 5
                done = True
                info = {"success": True}
                self.grid[self.pos[1],self.pos[0]] = 0
                self.grid[y,x] = AGENT
                self.pos = (x, y)
            elif (self.grid[y,x] == 0):
                self.grid[self.pos[1]][self.pos[0]] = 0
                self.grid[y][x] = AGENT
                self.pos = (x, y)

                cur_dis2goal = point_distance((x, y), self.goal_pos)
                if (cur_dis2goal < self.dis2goal):
                    reward += (self.dis2goal - cur_dis2goal)*1.5
                    self.dis2goal = cur_dis2goal
        else:
            reward -= 1
        reward -= 1

        self.step_count += 1
        if (self.step_count > self.max_steps):
            done = True

        return self.grid.copy(), reward, done, info

    def reset(self):
        self.grid, self.pos, self.goal_pos = self.generator.gene()
        self.h, self.w = self.grid.shape
        self.max_steps = self.h * self.w

        # need reset each time
        self.dis2goal = point_distance((0, 0), self.goal_pos)
        self.step_count = 0
        return self.grid.copy()
