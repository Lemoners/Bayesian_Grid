""" 
Visualize the distribution of grids sampled through VAE and compare to grids generated by 
MazeGene. (testing environment)
"""
from mygrid.MiniGrid.AE.visualize import visualize, test_solve

import sys
import os
import datetime

msavefig = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pictures", "vae")

if not os.path.exists(msavefig):
    os.makedirs(msavefig)

if __name__ == "__main__":
    time = datetime.datetime.now()
    time = datetime.datetime.strftime(time,'%m_%d_%H_%M_%S')
    msavefig += "/visualize_" + time + ".jpg"
    visualize(samples=1000, dim_reduction="pca", msavefig=msavefig)


