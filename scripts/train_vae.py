import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mygrid
import gym
from mygrid.MiniGrid.AE.VAE import train_vae, sample_vae

# train_vae()
print(sample_vae(1))




