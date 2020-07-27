from ..Discriminator.HardMazeDiscriminator import HardMazeDiscriminator
from .MazeGene import MazeGene
from .HyperPara import GRID_HEIGHT, GRID_WIDTH
import numpy as np
class HardMazeGene(object):
    def __init__(self):
        super().__init__()
        self.discriminator = HardMazeDiscriminator()
        self.generator = MazeGene()
    
    def gene(self, threshold=0.01):
        all_batches = int(1 / threshold)
        mazes = []
        scores = []
        for _ in range(all_batches):
            maze, _, _ = self.generator.gene()
            score, _ = self.discriminator.evaluate_maze(maze)
            mazes.append(maze)
            scores.append(score)
        ind = np.argmax(scores)
        mazes = np.array(mazes)[ind]
        return mazes, (0,0), (GRID_WIDTH-1, GRID_HEIGHT-1)


    def batch_gene(self, batches=1, threshold=0.1):
        """
        Generate `batches` of hard example, only use the hardest `threshold`
        """
        all_batches = int(batches / threshold)
        mazes = []
        scores = []
        for _ in range(all_batches):
            maze, _, _ = self.generator.gene()
            score, _ = self.discriminator.evaluate_maze(maze)
            mazes.append(maze)
            scores.append(score)
        ind = np.argpartition(scores, batches)[-batches:]
        mazes = np.array(mazes)[ind]
        return mazes






