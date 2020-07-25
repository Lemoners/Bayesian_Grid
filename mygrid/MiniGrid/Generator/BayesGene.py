from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from .HyperPara import Z_DIM
import numpy as np
from ..AE import VAE
import os
import torch
from .HyperPara import GRID_HEIGHT, GRID_WIDTH, AGENT, GOAL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BayesGene(object):
    """
    Describe: Using Bayesian Optimization, but fixed agent start && end point.
    """
    def __init__(self):
        self.X = []
        self.Y = []
        # poster sample number
        self.posterior_sample_number = 100
        # memory pointer
        self.position = 0

        # current parameter, numpy
        self.z = np.random.randn(Z_DIM)
        #kernel
        # K = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 50.0), nu=1.5)
        K = 1.0 * Matern(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=K)

        # model_dir
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + '/AE/model/VAE.pkl'
        self.vae = VAE()
        self.vae.load_state_dict(torch.load(model_dir))
        self.vae = self.vae.to(device).eval()

        self.capacity = 500
    
    def gene(self):
        grid = self._decode(torch.from_numpy(self.z).float().unsqueeze(0).to(device))
        grid = grid.cpu().round().detach().numpy().reshape(GRID_HEIGHT, GRID_WIDTH)
        grid[0, 0] = AGENT
        grid[GRID_HEIGHT-1, GRID_WIDTH-1] = GOAL
        return grid, (0,0), (GRID_HEIGHT-1, GRID_WIDTH-1)
    
    
    def update(self, data, z=[]):
        if (len(z) == 0):
            z = self.z
        self.push(z, data)
    
    def set_z(self, z):
        self.z = z

    def update_z(self):
        self.z = self.choose_next_sample()
    
    def _encode(self, x):
        mu, lvar = self.vae.encode(x)
        return self.vae.reparameterize(mu, lvar)
    
    def _decode(self, z):
        return self.vae.decode(z)
        
    def push(self, x, y):
        if len(self.X) < self.capacity:
            self.X.append(None)
            self.Y.append(None)
        self.X[self.position] = x
        self.Y[self.position] = y
        self.position = (self.position + 1) % self.capacity
    
    def _create_sample_x(self):
        samples = []
        for i in range(self.posterior_sample_number):
            s = np.random.randn(Z_DIM)
            samples.append(s)
        return samples
    
    def choose_next_sample(self):
        self.gp = self.gp.fit(self.X, self.Y)
        x_samples = self._create_sample_x()

        posterior_sample = self.gp.sample_y(x_samples, 1).T[0]
        which_max = np.argmax(posterior_sample)
        next_sample = x_samples[which_max]
        return next_sample