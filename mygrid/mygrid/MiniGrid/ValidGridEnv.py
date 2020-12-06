from .BasicGridEnv import BasicGridEnv
from .Generator import ValidGene

class ValidGridEnv(BasicGridEnv):
    """ ValidGridEnv

    Environment to generate solvable grids through randomly generating grids.
    """
    def __init__(self, generator=ValidGene()):
        super().__init__(generator=generator)
