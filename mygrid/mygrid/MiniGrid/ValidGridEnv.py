from .BasicGridEnv import BasicGridEnv
from .Generator import ValidGene

class ValidGridEnv(BasicGridEnv):
    def __init__(self, generator=ValidGene()):
        super().__init__(generator=generator)
