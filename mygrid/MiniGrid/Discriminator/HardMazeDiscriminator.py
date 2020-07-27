from ..Utils.BFS import BFSAgent
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HardMazeDiscriminator(object):
    def __init__(self):
        super().__init__()
        self.bfs = BFSAgent()
    
    def evaluate_maze(self, maze):
        """
        How hard the maze is:

        :return hard: [0,1]
        :return solvable: boolean
        """
        h, s = self.bfs.solve_with_distance(maze)
        if not s:
            return 0, s
        
        linear_h_start = h[0]
        linear_h_end = h[-1]
        linear_delta = (linear_h_end - linear_h_start) / (len(h) - 1)
        linear_h = [linear_h_start + linear_delta * int(i) for i in range(len(h))]

        h = torch.FloatTensor(h).to(device)
        linear_h = torch.FloatTensor(linear_h).to(device)
        h = F.softmax(h, dim=0)
        linear_h = F.softmax(linear_h, dim=0)
        kl_div = F.kl_div(linear_h, h, reduction="batchmean").cpu().numpy()
        return kl_div, s

        
        
        
