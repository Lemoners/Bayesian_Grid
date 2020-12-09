from mygrid.MiniGrid.Generator.HyperPara import DATA_BEFORE_UPDATE, RANDOM_DATA_BEFORE_UPDATE, RANDOM_PARA_BEFORE_UPDATE, GRID_HEIGHT, GRID_WIDTH, Z_DIM
from mygrid.MiniGrid.Utils.BFS import BFSAgent
from mygrid.MiniGrid.BayesGridEnv import BayesGridEnv, RandomBayesGridEnv
from mygrid.MiniGrid.Agent import ILNet
from mygrid.MiniGrid.Memory import ReplayBuffer
from mygrid.MiniGrid import BasicGridEnv, ValidGridEnv, MazeGridEnv
from mygrid.MiniGrid.HardGridEnv import HardGridEnv
from mygrid.MiniGrid.Generator.HardMazeGene import HardMazeGene
from mygrid.MiniGrid.Generator.MazeGene import MazeGene
from mygrid.MiniGrid.Discriminator.AgentDiscriminator import AgentDiscriminator

import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import random
import copy
from collections import namedtuple

# random_sample_rate = 10 # random_sample_rate / 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_save = os.path.dirname(os.path.abspath(__file__)) + "/data/model/IL"
if not os.path.exists(model_save):
    os.makedirs(model_save)

para_save = os.path.dirname(os.path.abspath(__file__)) + "/data/bayes_para"
if not os.path.exists(para_save):
    os.makedirs(para_save)


parser = argparse.ArgumentParser(description="Training IL Agent")
parser.add_argument('-e', '--envs', nargs='+', help='Environment for training.', default=\
    ['RandomBayesGridEnv', 'BayesGridEnv', 'BasicGridEnv', 'ValidGridEnv', 'MazeGridEnv'])
parser.add_argument('--num-epoches', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--demo-epoch', type=int, default=10)
parser.add_argument('--evaluate-epoch', type=int, default=50)
parser.add_argument('--pre-collect', type=int, default=5000)
parser.add_argument('-s', '--save-interval', type=int, default=20)
parser.add_argument('--random-sample-rate', type=int, default=10, help="Random sample rate is `random_sample_rate` percent.")
parser.add_argument('-l', '--log', default="")

def train(env, env_str, num_epoches=1000, batch_size=64, learning_rate=1e-3, demo_epoch=20, pre_collect=5000, save_interval=50, log_interval=20):
    """ Training IL agents with pure Behavior Cloning.

    Used for training IL agents in:
        RandomBayesGridEnv: z->VAE->grids, but without Bayesian Optimization.
        BasicGridEnv: random generated grids. (solvable or unsolvable)
        ValidGridEnv: random generated grids, but we ensure that they are solvable.
        MazeGridEnv: random generated maze-like grids through maze-generation algorithm.
    """
    model_state_dicts = {}
    model = ILNet().to(device)
    memory = ReplayBuffer()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bfs = BFSAgent()
    solvable_maze = 0
    bad_maze = 1

    maze_env = MazeGridEnv()

    # Pre_collect
    while (len(memory) < pre_collect):
        obs = env.reset()
        history, solvable = bfs.solve(obs)
        if solvable:
            for h, a in history:
                memory.push(torch.tensor(h), torch.tensor(a))

    for epoch in range(num_epoches):        
        data_loader = torch.utils.data.DataLoader(dataset=memory, batch_size=batch_size, shuffle=True)
        for i, (states, actions) in enumerate(data_loader):
            states = states.to(device)
            actions = actions.to(device)
            output_action_batch = model(states.float())
            loss = criterion(output_action_batch, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        _, score, _, _, _ = evaluate(maze_env, model)

        print ('\rEpoch [{}/{}], Loss: {:.4f}, Maze: {:.2f}, Score: {:.2f}'.format(epoch+1, num_epoches, loss.item(), solvable_maze / (solvable_maze + bad_maze), score), end="", flush=True)
        
        if (epoch+1) % log_interval == 0:
            print("")
        
        if (epoch+1) % save_interval == 0:
            model_state_dicts[str(epoch+1)] = copy.deepcopy(model.state_dict())
            torch.save(model_state_dicts, model_save + "/" + env_str)
        print("")
            

def bayes_train(env, env_str, num_epoches=1000, batch_size=64, learning_rate=1e-3, evaluate_epoch=150, demo_epoch=20, pre_collect=5000, save_interval=50, log_interval=20, random_sample_rate=10):
    """ Training IL with Behavior Cloning while updating the environment parameters with Bayesian Optimization.

    Used for training IL agents in:
        BayesGridEnv: z->VAE->grids, and with Bayesian Optmization.
    """
    model_state_dicts = {}
    model = ILNet().to(device)
    env = BayesGridEnv()
    Transition = namedtuple("Transition", ("para", "state", "action"))
    memory = ReplayBuffer(capacity=10000, memtype=Transition)

    history_grid = []

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bfs = BFSAgent()

    solvable_maze = 0
    bad_maze = 0

    agent_dis = AgentDiscriminator()
    hard_maze_env = HardGridEnv()
    maze_env = MazeGridEnv()

    for epoch in range(num_epoches):
        scores = []
        for _ in range(evaluate_epoch):
            is_random = False
            if (random.randint(1, 100) <= random_sample_rate):
                env.generator.set_z(env.generator.random())
                is_random = True
            else:
                env.generator.set_z(env.generator.sample())

            pre_score, pre_solve, history, _bad_maze, _solvable_maze = evaluate(env, model, episodes=5, history_grid=history_grid, need_expert_demo=True)

            if not is_random:
                bad_maze += _bad_maze
                solvable_maze += _solvable_maze

            if len(history) == 0:
                env.generator.update(env.generator.get_z(), -1)
            
            for h, a in history:
                memory.push(torch.tensor(env.generator.get_z()), torch.tensor(h), torch.tensor(a))

        if (len(memory) > batch_size):
            data_loader = torch.utils.data.DataLoader(dataset=memory, batch_size=batch_size, shuffle=False)
            for i, (paras, states, actions) in enumerate(data_loader):
                states = states.to(device)
                actions = actions.to(device)
                output_action_batch = model(states.float())
                loss = criterion(output_action_batch, actions)

                _loss = loss.clone()
                para_loss = zip(paras.cpu().detach().numpy(), _loss.cpu().detach().numpy())

                loss = torch.mean(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        # evaluate
        after_score, after_solve, _history, _, _ = evaluate(maze_env, model)

        print ('\rEpoch [{}/{}], Loss: {:.4f}, Maze: {:.2f}, Solve: {:.2f}, Memories: {}'.format(epoch+1, num_epoches, loss.item(), solvable_maze / (solvable_maze + bad_maze + 1), after_solve, len(memory)), end="", flush=True)
        
        env.generator.update_gp()

        if (epoch+1) % log_interval == 0:
            print ("")

        if (epoch+1) % save_interval == 0:
            model_state_dicts[str(epoch+1)] = copy.deepcopy(model.state_dict())
            with open(para_save + "/history" + env_str + ".json", "w+") as f:
                json.dump(history_grid, f)
            torch.save(model_state_dicts, model_save + "/" + env_str)

def evaluate(env, model, history_grid=None, episodes=50, need_expert_demo=False):
    model = model.to(device)
    model.eval()

    values = []
    solve = [0]
    history = []
    
    bfs = BFSAgent()
    bad_maze = 0
    added = 0

    for i in range(episodes):
        obs = env.reset()
        if need_expert_demo:
            # parameterized
            h, s = bfs.solve(obs)
            if s:
                while True:
                    action = model.predict(obs)
                    obs, reward, done, info = env.step(action)
                    if done:
                        if info.get('success'):
                            values.append(0.2)
                            solve.append(1)
                        else:
                            if len(history) == 0:
                                history.extend(h)
                                if not history_grid is None:
                                    history_grid.append(obs.tolist())
                            values.append(1)
                            solve.append(0)
                        break
            else:
                bad_maze += 1
                values.append(-1)
        

        else:
            while True:
                action = model.predict(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    if info.get('success'):
                        values.append(0.2)
                        solve.append(1)
                    else:
                        values.append(1)
                        solve.append(0)
                    break
    model.train()
    return np.mean(values), np.mean(solve), history, bad_maze, episodes-bad_maze


if __name__ == "__main__":
    args = parser.parse_args()
    envs = [eval(e) for e in args.envs]
    for i, _env in enumerate(envs):
        env = _env()
        print("Training IL Agent in {}".format(args.envs[i]))

        if 'BayesGridEnv' == args.envs[i]:
            bayes_train(env, env_str=args.envs[i]+args.log, num_epoches=args.num_epoches, batch_size=args.batch_size, learning_rate=args.learning_rate, evaluate_epoch=args.evaluate_epoch ,demo_epoch=args.demo_epoch, pre_collect=args.pre_collect, save_interval=args.save_interval, random_sample_rate=args.random_sample_rate)
        else:
            train(env, env_str=args.envs[i], num_epoches=args.num_epoches, batch_size=args.batch_size, learning_rate=args.learning_rate, demo_epoch=args.demo_epoch, pre_collect=args.pre_collect, save_interval=args.save_interval)