from .Generator.HardMazeGene import HardMazeGene
from .BasicGridEnv import BasicGridEnv

class HardGridEnv(BasicGridEnv):
    def __init__(self, generator=HardMazeGene()):
        super().__init__(generator=generator)



