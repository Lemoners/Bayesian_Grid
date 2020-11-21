from .HyperPara import *
import numpy as np
from ..Utils.BFS import BFSAgent


class ValidGene(object):
    def __init__(self):
        super().__init__()
        self.bfs = BFSAgent()

    def gene(self):
        """
        Describe: generate solvable maze
        return: grid: numpy.matrix
        return: pos: (0,0)
        return: goal_pos: (GRID_WIDTH-1,GRID_HEIGHT-1)
        """
        while True:
            grid = np.random.randint(0, 2, (GRID_HEIGHT, GRID_WIDTH))
            grid[0, 0] = AGENT
            grid[GRID_HEIGHT - 1, GRID_WIDTH - 1] = GOAL
            h, solvable = self.bfs.solve(grid)
            if solvable:
                break
        return grid, (0, 0), (GRID_WIDTH-1, GRID_HEIGHT-1)

    def update(self, data):
        pass

class SimpleValidGene(ValidGene):
    """
    Describe: gene maze only with wall and empty cell (used for VAE)
    return: grid
    """
    def gene(self):
        grid, pos, goal_pos = super().gene()
        grid[pos[1], pos[0]] = 0
        grid[goal_pos[1], goal_pos[0]] = 0
        return grid