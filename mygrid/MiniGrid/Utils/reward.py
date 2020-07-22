import numpy as np

def point_distance(p1, p2):
    def l1(x, y): return abs(x[0]-y[0]) + abs(x[1]-y[1])
    l1 = l1(p1, p2)
    l2 = np.linalg.norm(np.array(p1)-np.array(p2), ord=2)
    return l1
