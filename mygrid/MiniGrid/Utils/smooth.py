import numpy as np
def smooth(data, sm=1):
    if sm > 1:
        y = np.ones(sm)*1.0/sm
        data = np.convolve(y, data, "same")
    return data

def average_pooling(data, av=2):
    if(av > 1):
        y = data.reshape(-1, av)
        data = np.average(y, axis=1)
    return data

def conv2d_size_out(size, kernel_size=2, stride=1, padding=2):
    return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1