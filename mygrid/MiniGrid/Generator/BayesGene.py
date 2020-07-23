from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from .HyperPara import *

class BayesGene(object):
    def __init__(self):
        self.X = []
        self.Y = []
        # poster sample number
        self.posterior_sample_number = 100
        # memory pointer
        self.position = 0
        
        #kernel
        K = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
        self.gp = GaussianProcessRegressor(kernel=K)
        
        self.capacity = 10000
        
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
            s = np.random.rand(BAYES_X)*2*OMEGA_RANGE - OMEGA_RANGE
            samples.append(s)
        return samples
    
    def choose_next_sample(self):
        self.gp = self.gp.fit(self.X, self.Y)
        # 转置
        x_samples = self._create_sample_x()
        posterior_sample = self.gp.sample_y(x_samples, 1).T[0]
        which_max = np.argmax(posterior_sample)
        next_sample = x_samples[which_max]
        return next_sample
        