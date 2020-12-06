from .BasicGridEnv import BasicGridEnv
from .Generator.BayesGene import BayesGene 
from .Generator.HyperPara import DATA_BEFORE_UPDATE, RANDOM_DATA_BEFORE_UPDATE, RANDOM_PARA_BEFORE_UPDATE, Z_DIM
from .Utils import smooth, average_pooling
from .Utils.BFS import BFSAgent
from .Generator.MazeGene import MazeGene
import numpy as np


class BayesGridEnv(BasicGridEnv):
    """ BayesGridEnv

    Environment to generate parametric grids with Bayesian Optimization and VAE.
    """
    def __init__(self, generator=BayesGene()):
        super(BayesGridEnv, self).__init__(generator=generator)
        self.bfs = BFSAgent()
        self.maze_gene = MazeGene()
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

    def reset(self):
        grid = super().reset()
        return grid

class RandomBayesGridEnv(BayesGridEnv):
    def __init__(self, generator=BayesGene()):
        super().__init__(generator=generator)
    
    def reset(self):
        self.generator.set_z(self.generator.random())
        grid = super().reset()
        return grid

