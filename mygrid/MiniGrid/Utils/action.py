import numpy as np

def direction2action(delta_x, delta_y):
    if (delta_x == 0):
        if (delta_y == 1):
            return 2
        elif (delta_y == 0):
            return 0
        elif (delta_y == -1):
            return 1
    elif (delta_x == 1 and delta_y == 0):
        return 4
    elif (delta_x == -1 and delta_y == 0):
        return 3
    raise Exception("Unknown direction")


def action2direction(action):
    a2d = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
    return a2d[action]


def find_obj(grid, obj):
    y, x = np.where(grid==obj)
    if (len(y) > 0):
        return (x[0], y[0])
    else:
        return (-1, -1)