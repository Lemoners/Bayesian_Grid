from .HyperPara import *
import numpy as np
from ..Utils.BFS import BFSAgent


class ValidGene(object):
    def __init__(self):
        super().__init__()
        self.bfs = BFSAgent()

    def gene(self):
        """ Generate solvable grids through random sampling.

        :returns: grid: numpy.matrix
        :returns: start position for the agent: (0,0)
        :returns: goal position: (GRID_WIDTH-1,GRID_HEIGHT-1)
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
    """ Generate maze without agent and goal. (only with wall and empty cells, used for training VAE)
    return: grid
    """
    def gene(self):
        grid, pos, goal_pos = super().gene()
        grid[pos[1], pos[0]] = 0
        grid[goal_pos[1], goal_pos[0]] = 0
        return grid