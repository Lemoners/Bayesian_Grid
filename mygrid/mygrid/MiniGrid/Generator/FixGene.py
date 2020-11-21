from .MazeGene import MazeGene
import numpy as np

class FixGene(object):
    def __init__(self, size=10):
        self.gene = MazeGene()
        self.size = size
        self.para = np.arange(self.size)
        self.mazes = []
        for _ in range(size):
            self.mazes.append(self.gene.gene())
    
    def gene(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.size)
        return self.mazes[index]







