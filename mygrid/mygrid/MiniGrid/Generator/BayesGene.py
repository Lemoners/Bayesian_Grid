import numpy as np
from ..AE import VAE
import os
import torch
from .HyperPara import GRID_HEIGHT, GRID_WIDTH, AGENT, GOAL, Z_DIM
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import normalize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.neighbors import KDTree
from .MazeGene import MazeGene
from ..Utils.BFS import BFSAgent
from ..Utils.AlphaSet import AlphaSet
import random
np.random.seed(0)

class BayesGene(object):
    """
    Describe: Using Bayesian Optimization, but fixed agent start && end point.
    """
    def __init__(self):
        # poster sample number
        self.posterior_sample_number = 500
        self.capacity = 250
        self.minimum_update_data = 10
        self.reset()
        # model_dir
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + '/AE/model/VAE.pkl'
        self.vae = VAE()
        self.vae.load_state_dict(torch.load(model_dir, map_location=device))
        self.vae = self.vae.to(device)
        self.vae.eval()
        self.maze_gene = MazeGene()
        self.bfs = BFSAgent()

    def gene(self):
        grid = self._decode(torch.from_numpy(self.z).float().unsqueeze(0).to(device))
        grid = grid.cpu().round().detach().numpy().reshape(GRID_HEIGHT, GRID_WIDTH).astype(np.uint8)
        grid[0, 0] = AGENT
        grid[GRID_HEIGHT-1, GRID_WIDTH-1] = GOAL
        return grid, (0,0), (GRID_WIDTH-1, GRID_HEIGHT-1)

    def update(self, z, data):
        self.memory.add(z, data)
    
    # def query_nn(self, z=[]):
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
        print(" Update GP", len(self.memory))
        if (len(self.memory) > self.minimum_update_data):
            self.choose_next_gp()
            self.z = self.sample()
    
    def sample(self, num=-1):
        if num == -1:
            num = self.posterior_sample_number

        x_samples = self._create_sample_x(num=self.posterior_sample_number)
        try:
            posterior_sample = self.gp.sample_y(x_samples, 1, random_state=None).T[0]
            which_max = np.argmax(posterior_sample)
            next_sample = x_samples[which_max]
        except Exception as e:
            next_sample = self.random()
            print(e)
        return next_sample

    def random(self):
        return np.random.randn(Z_DIM)

    def _encode(self, x):
        mu, lvar = self.vae.encode(x)
        return self.vae.reparameterize(mu, lvar)
    
    def _decode(self, z):
        return self.vae.decode(z)
        
    def _create_sample_x(self, num):
        return np.random.randn(num, Z_DIM)
    
    def choose_next_gp(self):
        # fit_y = (np.array(self.Y) - np.mean(self.Y))
        # n_var = np.var(self.Y)
        # if n_var != 0:
        #     fit_y /= n_var
        self.memory.calculate_mean()
        fit_z, fit_score = self.memory.get_data()
        # print(fit_z, fit_score)
        # input("choose_next_gp")
        fit_score = preprocessing.scale(fit_score)
        self.gp = self.gp.fit(fit_z, fit_score)
        self.memory.save_status()
        # assert 1 == 0

    def reset(self):
        # self.Z_LP = []
        # self.LP = []
        # self.Z_ALP = []
        # self.ALP = []
        # self.LP_position = 0
        # self.ALP_position = 0
        self.memory = AlphaSet(func=lambda new, old: old - new, capacity=self.capacity, dtype=np.float64)

        # current parameter, numpy
        # self.z = np.random.randn(Z_DIM)
        self.z = np.zeros(Z_DIM)

        K = 5.0 * Matern(length_scale=0.1)
        # K = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
            # + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        # K = C(1.0, (1e-3, 1e-1)) * RBF(10, (1e-2, 1e-1))
        # self.gp = GaussianProcessRegressor(kernel=K, normalize_y=True)
        # self.gp = GaussianProcessRegressor(kernel=K)
        self.gp = GaussianProcessRegressor(kernel=K, optimizer=None)