from ..Generator.HyperPara import *
from .action import direction2action, find_obj
import numpy as np


class BFSAgent(object):
    def solve(self, maze):
        """
        param: maze: numpy.matrix
        return: history: [numpy.matrix, action]
        return: solvable: boolean
        """

        # reset param
        self._reset(maze)

        self.search_maze = maze.copy()
        self.visited = maze.copy()
        self.visited[np.where(self.visited == GOAL)] = 0
        self.visited[np.where(self.visited == AGENT)] = 0

        search = []
        history = []

        # start BFS
        search.append((0, 0))
        self.visited[0, 0] = 1

        while(len(search) > 0):
            x, y = search[0]
            search = search[1:]

            if (self.search_maze[y, x] == GOAL):
                self.search_maze[np.where(self.search_maze == AGENT)] = 0
                hx, hy = x, y
                while True:
                    if (hx == 0 and hy == 0):
                        break
                    thisx, thisy = self.parent[hy, hx]
                    action = direction2action(hx-thisx, hy-thisy)
                    if (self.search_maze[hy, hx] == AGENT):
                        self.search_maze[hy, hx] = 0
                    self.search_maze[thisy, thisx] = AGENT
                    hx, hy = thisx, thisy
                    history.append((self.search_maze.copy(), action))
                break
            neibs = self._get_neighbour((x, y))
            for n in neibs:
                nx, ny = n
                self.visited[ny, nx] = 1
                self.parent[ny, nx] = (x, y)
                search.append(n)

        history.reverse()
        solvable = True
        if len(history) == 0:
            solvable = False
        return history, solvable

    def _get_neighbour(self, pos):
        h, w = self.search_maze.shape
        x, y = pos
        delta_x = [0, 0, -1, 1]
        delta_y = [-1, 1, 0, 0]
        neibs = []

        for i in range(4):
            new_x = x + delta_x[i]
            new_y = y + delta_y[i]
            if (0 <= new_x < w and 0 <= new_y < h and self.visited[new_y][new_x] == 0):
                neibs.append((new_x, new_y))
        return neibs

    def _reset(self, maze):
        self.parent = np.array([[(-1, -1) for i in range(maze.shape[1])]
                                for i in range(maze.shape[0])])
        self.pos = find_obj(maze, AGENT) 
        self.h, self.w = maze.shape
