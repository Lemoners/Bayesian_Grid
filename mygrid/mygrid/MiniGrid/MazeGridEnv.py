from .Generator import MazeGene
from .BasicGridEnv import BasicGridEnv

class MazeGridEnv(BasicGridEnv):
    """ MazeGridEnv

    Environment to generate solvable maze-like grids.
    """
    def __init__(self, generator=MazeGene()):
        super().__init__(generator=generator)


