import mygrid.MiniGrid
from gym.envs.registration import register

register(
    id='basicgrid-v0',
    entry_point='mygrid.MiniGrid:BasicGridEnv',
)

register(
    id='validgrid-v0',
    entry_point='mygrid.MiniGrid:ValidGridEnv'
)

register(
    id='mazegrid-v0',
    entry_point='mygrid.MiniGrid:MazeGridEnv'
)