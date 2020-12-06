from ..AE import VAE
from .HyperPara import GRID_HEIGHT, GRID_WIDTH, AGENT, GOAL, Z_DIM
from .MazeGene import MazeGene
from ..Utils.BFS import BFSAgent
from ..Utils.AlphaSet import AlphaSet

import numpy as np
import os
import torch
import random
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree

np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BayesGene(object):
    """ BayesGene
    
    Generate parametric grids with Bayesian Optimization (Gaussian Process and Thompson Sampling) and VAE.
    """
    def __init__(self):
        self.posterior_sample_number = 500
        self.capacity = 250
        self.minimum_update_data = 10
        self.reset()
        
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + '/AE/model/VAE.pkl'
        self.vae = VAE()
        self.vae.load_state_dict(torch.load(model_dir, map_location=device))
        self.vae = self.vae.to(device)
        self.vae.eval()
        
        self.maze_gene = MazeGene()
        self.bfs = BFSAgent()

    def gene(self):
        """ Generate grid with current parameter (self.z) through VAE.

        :returns: grid: numpy.matrix
        :returns: start position for the agent: (0,0)
        :returns: goal position: (GRID_WIDTH-1,GRID_HEIGHT-1)
        """
        grid = self._decode(torch.from_numpy(self.z).float().unsqueeze(0).to(device))
        grid = grid.cpu().round().detach().numpy().reshape(GRID_HEIGHT, GRID_WIDTH).astype(np.uint8)
        grid[0, 0] = AGENT
        grid[GRID_HEIGHT-1, GRID_WIDTH-1] = GOAL
        return grid, (0,0), (GRID_WIDTH-1, GRID_HEIGHT-1)

    def update(self, z, data):
        """ Update the parameter-score pair (z, data) into self.memory pool.
        """
        self.memory.add(z, data)
    
    # def query_nn(self, z=[]):
    #     """ Query the neareast neighbor. (Euclidean distance)
    #     """
    #     if len(z) == 0:
    #         z = self.z.copy()
    #     tree = KDTree(np.array(self.memory.keys), leaf_size=2)                    
    #     dist, ind = tree.query(z.reshape(1, -1), k=1)     
    #     return(np.array(self.memory.keys)[ind])  # indices of 3 closest neighbors
    
    def set_z(self, z):
        self.z = z.copy()

    def get_z(self):
        return self.z.copy()

    def update_gp(self):
        """ Fit Gaussian Process to the memory pool and update current parameter self.z.
        """
        print(" Update GP", len(self.memory))
        if (len(self.memory) > self.minimum_update_data):
            self.choose_next_gp()
            self.z = self.sample(self.posterior_sample_number)
    
    def sample(self, num=500):
        """ Generate a sample from current Gaussian Process through Thompson Sampling.  
        """
        x_samples = np.random.randn(num, Z_DIM)

        try:
            posterior_sample = self.gp.sample_y(x_samples, 1, random_state=None).T[0]
            which_max = np.argmax(posterior_sample)
            next_sample = x_samples[which_max]
        except Exception as e:
            next_sample = self.random()
            print(e)
        return next_sample

    def random(self):
        """ Return a random parameter.
        """
        return np.random.randn(Z_DIM)

    def _encode(self, x):
        """ Encode grid into hidden parameters with VAE.
        """
        mu, lvar = self.vae.encode(x)
        return self.vae.reparameterize(mu, lvar)
    
    def _decode(self, z):
        """ Decode grid from hidden parameters with VAE.
        """
        return self.vae.decode(z)
        
    def choose_next_gp(self):
        """ Fit Gaussian Process to the memory pool.
        """
        self.memory.calculate_mean()
        fit_z, fit_score = self.memory.get_data()
        fit_score = preprocessing.scale(fit_score)
        self.gp = self.gp.fit(fit_z, fit_score)
        self.memory.save_status()

    def reset(self):
        """ Reset.
        """
        self.memory = AlphaSet(func=lambda new, old: old - new, capacity=self.capacity, dtype=np.float64)

        # Reset current parameter.
        self.z = np.zeros(Z_DIM)

        # Kernel function for Gaussian Process.
        K = 5.0 * Matern(length_scale=0.1)

        # K = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
            # + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        # K = C(1.0, (1e-3, 1e-1)) * RBF(10, (1e-2, 1e-1))
        
        # self.gp = GaussianProcessRegressor(kernel=K)
        self.gp = GaussianProcessRegressor(kernel=K, optimizer=None)