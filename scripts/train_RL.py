"""Train RL agents using DQN with tianshou (https://github.com/thu-ml/tianshou)"""

import os
import sys
import tianshou as ts
import torch, numpy as np
from torch import nn

from mygrid.MiniGrid import MazeGridEnv, SparseRewardWrapper, RenderWrapper
from mygrid.MiniGrid.Utils import conv2d_size_out
from mygrid.MiniGrid.Generator.HyperPara import GRID_HEIGHT, GRID_WIDTH

import argparse

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'model','RL','DQN')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


env = MazeGridEnv()
train_envs = ts.env.DummyVectorEnv([lambda: MazeGridEnv() for _ in range(8)])
test_envs = ts.env.DummyVectorEnv([lambda: SparseRewardWrapper(MazeGridEnv()) for _ in range(100)])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--discount-factor', type=float, default=0.9)
parser.add_argument('--buffer-size', type=int, default=20000)
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)

class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        w = conv2d_size_out(conv2d_size_out(GRID_WIDTH))
        h = conv2d_size_out(conv2d_size_out(GRID_HEIGHT))
        self.model = nn.Sequential(*[
            nn.Conv2d(1, 8, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ])
        self.fc = nn.Linear(h*w*16, np.prod(action_shape))
    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float).to(device)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, 1, GRID_HEIGHT, GRID_WIDTH))
        logits = self.fc(logits.view(batch, -1))
        return logits, state

def train_RL(lr=1e-3, discount_factor=0.9, buffer_size=20000, max_epoch=100, batch_size=64):
    """ Train RL agent using DQN with tianshou (https://github.com/thu-ml/tianshou)

    :args lr: learning rate.
    :args discount_factor: discount_factor w.r.t. cumulated reward.
    :args buffer_size: replay buffer size for DQN.
    :args max_epoch: max training epoch.
    :args batch_size: batch size.

    """
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    policy = ts.policy.DQNPolicy(net, optim,
        discount_factor=discount_factor, estimation_step=3,target_update_freq=320)

    train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(size=buffer_size))
    test_collector = ts.data.Collector(policy, test_envs)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=max_epoch, step_per_epoch=1000, collect_per_step=10,
        episode_per_test=100, batch_size=batch_size,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda x: x >= 0.8,
        writer=None)

    print(f'Finished training! With {result["duration"]}')
    torch.save(policy.state_dict(), save_dir + '/dqn.pth')

if __name__ == "__main__":
    args = parser.parse_args()
    train_RL(lr=args.lr, discount_factor=args.discount_factor, buffer_size=args.buffer_size, max_epoch=args.max_epoch, batch_size=args.batch_size)