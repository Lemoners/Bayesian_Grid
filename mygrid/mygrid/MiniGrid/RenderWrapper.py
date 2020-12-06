import tkinter as tk
import sys
import random
from .Generator.HyperPara import GRID_HEIGHT, GRID_WIDTH, AGENT, GOAL
from .Utils import find_obj
import numpy as np
import time

class RenderWrapper(tk.Frame):
    """ RenderWrapper
    
    Wrapper to render the environment.
    """
    def __init__(self, env, width=GRID_WIDTH, height=GRID_HEIGHT, size=50):
        super(RenderWrapper, self).__init__()
        self.env = env
        self.pos = (0, 0)
        self.mwidth, self.mheight = width, height
        self.msize = size
        self.grid()
        self._create_widgets()
    
    def _create_widgets(self):
        w = self.mwidth * self.msize
        h = self.mheight * self.msize
        self.canvas = tk.Canvas(self, width=w, height=h)
        self.canvas.grid()

    def _draw_maze(self, maze, pos, goal_pos):
        self.canvas.delete('all')
        for x in range(self.mwidth):
            for y in range(self.mheight):
                x0 = x * self.msize
                y0 = y * self.msize
                color = ""
                if ((x,y) == pos):
                    color = "blue"
                elif ((x,y) == goal_pos):
                    color = "green"
                elif (maze[y,x] == 1):
                    color = "black"
                if (color != ""):
                    id = self.canvas.create_rectangle(x0, y0, x0+self.msize, y0+self.msize, width=0, fill=color)
                if color == "blue":
                    self.agent = id
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        ny, nx = find_obj(obs, AGENT)
        lx, ly = self.pos
        self.pos = (nx, ny)
        self.canvas.move(self.agent, (int)(ny-ly)*self.msize, (int)(nx-lx)*self.msize)
        self.update()
        time.sleep(0.1)
        return obs, reward, done, info
    
    
    def reset(self):
        obs = self.env.reset()
        pos_x, pos_y = find_obj(obs, AGENT)
        goal_x, goal_y = find_obj(obs, GOAL)
        self.pos = (pos_x, pos_y)
        self._draw_maze(obs, (pos_x, pos_y), (goal_x, goal_y))
        self.update()
        return obs




        


