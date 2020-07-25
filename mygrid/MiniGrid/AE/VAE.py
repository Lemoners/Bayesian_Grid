import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..Generator import MazeGene
from ..Generator.HyperPara import GRID_HEIGHT, GRID_WIDTH


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model save
model_save = os.path.abspath(__file__)
if not os.path.exists(model_save):
    os.mkdir(model_save)

# Hyper-parameters
GRID_SIZE = GRID_HEIGHT * GRID_WIDTH
h_dim = 40
z_dim = 4
num_epochs = 25
batch_size = 64
learning_rate = 1e-3

# data gene
gene = MazeGene()


# VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(GRID_SIZE, h_dim)
        self.fmu = nn.Linear(h_dim, z_dim)
        self.flog_var = nn.Linear(h_dim, z_dim)
        self.f2 = nn.Linear(z_dim, h_dim)
        self.f3 = nn.Linear(h_dim, GRID_SIZE)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fmu(h), self.flog_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc2(z))
        return F.sigmoid()

















