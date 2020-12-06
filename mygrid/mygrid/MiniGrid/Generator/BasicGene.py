from .HyperPara import *
import numpy as np


class BasicGene(object):
    """ BasicGene

    Generate random grids. (solvable or unsolvable)
    """
    def __init__(self):
        super().__init__()

    def gene(self):
        """ Gene grid.
        
        :returns: grid: numpy.matrix
        :returns: start position for the agent: (0,0)
        :returns: goal position: (GRID_WIDTH-1,GRID_HEIGHT-1)
        """
        grid = np.random.randint(0, 2, (GRID_HEIGHT, GRID_WIDTH))
        grid[0, 0] = AGENT
        grid[GRID_HEIGHT - 1, GRID_WIDTH - 1] = GOAL
        return grid, (0,0), (GRID_WIDTH-1, GRID_HEIGHT-1)

    def update(self, data):
        pass








