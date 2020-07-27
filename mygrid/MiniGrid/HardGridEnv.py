from .Generator.HardMazeGene import HardMazeGene
from .BasicGridEnv import BasicGridEnv

class HardGridEnv(BasicGridEnv):
    def __init__(self, generator=HardGene()):
        super().__init__(generator=generator)



