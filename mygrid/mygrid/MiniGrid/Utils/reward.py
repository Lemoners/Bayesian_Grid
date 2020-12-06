import numpy as np


def point_distance(p1, p2, norm="l1"):
    """ Calculate the distance between points in l1 or l2 norm.

    Can be used for calculate reward.

    :args p1: point1.
    :args p2: point2.
    :returns: distance under given norm.
    """
    def l1(x, y): 
        return abs(x[0]-y[0]) + abs(x[1]-y[1])
    dis = 0
    if (norm == "l1"):
        dis = l1(p1, p2)
    elif (norm == "l2"):
        dis = np.linalg.norm(np.array(p1)-np.array(p2), ord=2)
    return dis