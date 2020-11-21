from .Generator import MazeGene
from .BasicGridEnv import BasicGridEnv

class MazeGridEnv(BasicGridEnv):
    def __init__(self, generator=MazeGene()):
        super().__init__(generator=generator)


