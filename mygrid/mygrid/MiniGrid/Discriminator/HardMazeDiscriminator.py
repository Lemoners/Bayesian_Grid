from ..Utils.BFS import BFSAgent
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HardMazeDiscriminator(object):
    """ HardMazeDiscriminator

    Discriminator for evaluate the diffculty of given grid based on hand-crafted metric.
    For any given grid, we first solve it to get the sequence of `s=[distance_between_the_agent_and_the_goal]` during solving.
    Then we calculate the kl_divergence between `s` and a linear_decrease_sequence with the same start point, end point and length of `s`.
    The intuition is: if the agent has to make some detours before reaching the goal, then the maze is considered relatively difficult.
    And the detours can be measured by the kl_div between `s` and linear_decrease_sequence.
    """
    def __init__(self):
        super().__init__()
        self.bfs = BFSAgent()
    
    def evaluate_maze(self, maze):
        """ Evaluate the difficulty of the maze.

        :args maze: maze for evaluation.
        :returns: difficulty in [0.0, 1.0].
        :returns: solvable.
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
        