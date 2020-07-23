import numpy as np
import random
from .HyperPara import *
from ..Utils import action2direction

def OPPOSITE(direction):
    if (direction == 1):
        return 2
    if (direction == 2):
        return 1
    if (direction == 3):
        return 4
    if (direction == 4):
        return 3


class MazeGene(object):
    def gene(self):
        self._reset()
        self._carve_passage_from(0, 0)
        # print(self.gene_grid)
        self.grid = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        for cx in range(MAZE_WIDTH):
            for cy in range(MAZE_HEIGHT):
                self.grid[2*cy, 2*cx] = 0
        for cx in range(MAZE_WIDTH):
            for cy in range(MAZE_HEIGHT):
                for i in range(4):
                    if (self.gene_grid[cy, cx] & (1 << i) == (1 << i)):
                        direction = action2direction(i + 1)
                        nx, ny = 2 * cx + direction[0], 2 * cy + direction[1]
                        self.grid[ny, nx] = 0
        self.grid[self.pos] = AGENT
        self.grid[self.goal_pos] = GOAL

        return self.grid.copy(), self.pos, self.goal_pos

    def _carve_passage_from(self, cx, cy):
        # Up, Down, Left, Right
        directions = [1, 2, 3, 4]
        random.shuffle(directions)
        for d in directions:
            nx, ny = cx + action2direction(d)[0], cy + action2direction(d)[1]
            if (0 <= nx < MAZE_WIDTH and 0 <= ny < MAZE_HEIGHT and self.gene_grid[ny, nx] == 0):
                # print("Action {} is valid".format(d))
                self.gene_grid[cy, cx] |= (1 << (d - 1))
                self.gene_grid[ny, nx] |= (1 << (OPPOSITE(d) - 1))
                self._carve_passage_from(nx, ny)

    def _reset(self):
        # 0000 -> Up, Down, Left, Right
        self.gene_grid = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=np.uint8)
        self.pos = (0, 0)
        self.goal_pos = (np.random.randint(GRID_HEIGHT),
                        np.random.randint(GRID_WIDTH))

    def update(self, data):
        pass
