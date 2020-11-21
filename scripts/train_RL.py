import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tianshou as ts
import torch, numpy as np
from torch import nn
from mygrid.MiniGrid import MazeGridEnv, SparseRewardWrapper, RenderWrapper
from mygrid.MiniGrid.Utils import conv2d_size_out
from mygrid.MiniGrid.Generator.HyperPara import GRID_HEIGHT, GRID_WIDTH

env = MazeGridEnv()
train_envs = ts.env.VectorEnv([lambda: MazeGridEnv() for _ in range(8)])
test_envs = ts.env.VectorEnv([lambda: SparseRewardWrapper(MazeGridEnv()) for _ in range(100)])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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



state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape).to(device)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = ts.policy.DQNPolicy(net, optim,
    discount_factor=0.9, estimation_step=3,
    use_target_network=True, target_update_freq=320)

policy.load_state_dict(torch.load('dqn.pth'))

train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(size=20000))
test_collector = ts.data.Collector(policy, test_envs)


result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=100, step_per_epoch=1000, collect_per_step=10,
    episode_per_test=100, batch_size=64,
    train_fn=lambda e: policy.set_eps(0.1),
    test_fn=lambda e: policy.set_eps(0.05),
    stop_fn=lambda x: x >= 0.8,
    writer=None)

print(f'Finished training! Use {result["duration"]}')
torch.save(policy.state_dict(), 'dqn.pth')



# policy.load_state_dict(torch.load('dqn.pth'))
# policy.set_eps(0.0001)
# env = RenderWrapper(env)
# collector = ts.data.Collector(policy, env)
# data = collector.collect(n_episode=1)
# print(data)
# collector.close()






































# from stable_baselines3 import A2C, PPO
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3.common.cmd_util import make_vec_env
# from stable_baselines3.common.utils import set_random_seed


# SPARSE = False

# model_save = os.path.dirname(os.path.abspath(__file__)) + "/model/RL/PPO"

# def make_env(rank, seed=0):
#     """
#     Utility function for multiprocessed env.
#     """
#     def _init():
#         env = MazeGridEnv()
#         if SPARSE:
#             env = SparseRewardWrapper(env)
#         env.seed(seed + rank)
#         return env
#     set_random_seed(seed)
#     return _init

# if __name__ == '__main__':
#     num_cpu = 4  # Number of processes to use
#     # Create the vectorized environment
#     env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

#     # Stable Baselines provides you with make_vec_env() helper
#     # which does exactly the previous steps for you:
#     # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

#     time_steps = 200000
#     model = A2C('MlpPolicy', env, verbose=1)
#     model.learn(total_timesteps=time_steps)

#     if not SPARSE:
#         model_save += "/MazeGridEnv"
#     else:
#         model_save += "/SparseMazeGridEnv"

#     if not os.path.exists(model_save):
#         os.makedirs(model_save)

#     model.save(model_save + "/{}".format(time_steps))