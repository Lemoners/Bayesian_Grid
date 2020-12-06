from .Generator.HardMazeGene import HardMazeGene
from .BasicGridEnv import BasicGridEnv

class HardGridEnv(BasicGridEnv):
    """ HardGridEnv

    Environment to generate solvable and relatively hard grids. (with heuristic metric)
    """
    def __init__(self, generator=HardMazeGene()):
        super().__init__(generator=generator)



