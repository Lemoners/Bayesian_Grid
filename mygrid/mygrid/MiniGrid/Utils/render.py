from gym_minigrid.minigrid import *
from mygrid.MiniGrid.Generator.MazeGene import MazeGene
import numpy as np
from gym_minigrid.register import register

class MazeEnv(MiniGridEnv):
    """
    Grid environment with MazeGeneration algorithm.
    """
    def __init__(
        self,
        size=9,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty wall and surrounding walls.
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # gene = MazeGene()
        gene = HardMazeGene()

        grid, pos, goal_pos = gene.gene(grid_height=height-2, grid_width=width-2)

        # Place the goal at the bottom-right corner
        goal_x, goal_y = goal_pos
        self.put_obj(Goal(), goal_x + 1, goal_y + 1)

        # Place the agent
        pos_x, pos_y = pos
        self.agent_pos = (pos_x, pos_y)
        self.agent_dir = 0


        xs, ys = np.where(grid==1)
        for i in range(len(xs)):
            self.put_obj(Wall(), xs[i]+1, ys[i]+1)

        self.mission = "get to the green goal square"

class MazeEnv9x9(MazeEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, **kwargs)
    
    def _reward(self):
        return 10


class RenderMazeEnv9x9(MazeEnv):
    def __init__(self, size=9, _grid=None,**kwargs):
        self._grid = _grid
        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        # Create an empty wall and surrounding walls.
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        goal_y, goal_x = height - 2, width - 2
        pos_y, pos_x = 1, 1

        # Place the goal at the bottom-right corner
        self.put_obj(Goal(), goal_x, goal_y)

        # Place the agent
        self.agent_pos = (pos_x, pos_y)
        self.agent_dir = 0

        xs, ys = np.where(self._grid==1)
        for i in range(len(xs)):
            if (xs[i]+1, ys[i]+1) != (pos_x, pos_y) and (xs[i]+1, ys[i]+1) != (goal_x, goal_y):
                self.put_obj(Wall(), xs[i]+1, ys[i]+1)

        self.mission = "get to the green goal square"







