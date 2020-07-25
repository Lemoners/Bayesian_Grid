import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from ..Utils import conv2d_size_out
from ..Generator.HyperPara import GRID_HEIGHT, GRID_WIDTH

ACTION_SPACE_DIM = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ILNet(nn.Module):
    def __init__(self):
        super(ILNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        w = conv2d_size_out(conv2d_size_out(GRID_WIDTH))
        h = conv2d_size_out(conv2d_size_out(GRID_HEIGHT))

        self.fc = nn.Linear(h*w*16, ACTION_SPACE_DIM)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = f.softmax(out, dim=1)
        return out
    
    def predict(self, obs):
        obs = obs[np.newaxis, :, :]
        obs = torch.from_numpy(obs).unsqueeze(0).float()
        obs = obs.to(device)
        action = torch.argmax(self.forward(obs), dim=1)
        return action.item()









