import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..Generator.MazeGene import SimpleMazeGene
from ..Generator.HyperPara import GRID_HEIGHT, GRID_WIDTH, Z_DIM
from ..Utils import conv2d_size_out

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model save
model_save = os.path.dirname(os.path.abspath(__file__)) + "/model"
if not os.path.exists(model_save):
    os.mkdir(model_save)

# Hyper-parameters
GRID_SIZE = GRID_HEIGHT * GRID_WIDTH
h_dim = 25
z_dim = Z_DIM
data_size = 10000
num_epochs = 1000
update_iter = 1
batch_size = 128
learning_rate = 1e-3

# data gene
gene = SimpleMazeGene()


# VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.AvgPool2d(kernel_size=2, stride=2)
        # )
        # w = conv2d_size_out(GRID_WIDTH, 2, 1, 2) // 2
        # h = conv2d_size_out(GRID_HEIGHT, 2, 1, 2) // 2
        self.fc1 = nn.Linear(GRID_SIZE, h_dim)
        self.fmu = nn.Linear(h_dim, z_dim)
        self.flog_var = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, GRID_SIZE)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fmu(h), self.flog_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstruct = self.decode(z)
        return x_reconstruct, mu, log_var
    
def train_vae():
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def gene_data():
        data = []
        for i in range(data_size):
            print("\r Generating data: ", int(10*i/data_size)*'*', int(100*i/data_size), "%", end="", flush=True)
            data.append(gene.gene())
        print('\n Data generated')
        return data

    for it in range(update_iter):
        data = gene_data()
        data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

        # Start training
        for epoch in range(num_epochs):
            for i, x in enumerate(data_loader):
                # Forward pass
                x = x.to(device).view(-1, GRID_SIZE).float()
                x_reconstruct, mu, log_var = model(x)

                # reconstruct_loss = F.binary_cross_entropy(x_reconstruct, x, size_average=False)
                reconstruct_loss = F.binary_cross_entropy(x_reconstruct, x, reduction='sum')
                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                # back prop and optim
                loss = reconstruct_loss + kl_div
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 50 == 0:
                    print ("\r Iter [{}/{}], Epoch [{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}".format(it+1, update_iter, epoch+1, num_epochs, i+1, len(data_loader), reconstruct_loss.item(), kl_div.item()), end=" ", flush=True)
                    torch.save(model.state_dict(), model_save + "/VAE.pkl")
        print("")
    # save model
    torch.save(model.state_dict(), model_save + "/VAE.pkl")

def sample_vae(sample_size=batch_size):
    model = VAE()
    model.load_state_dict(torch.load(model_save + "/VAE.pkl"))
    model = model.to(device).eval()
    with torch.no_grad():
        z = torch.randn(sample_size, z_dim).to(device)
        out = model.decode(z).view(-1, GRID_HEIGHT, GRID_WIDTH).cpu().round().numpy()
    return out
















