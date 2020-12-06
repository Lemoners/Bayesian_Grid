import sys
import os
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
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import random
import copy
# import warnings
# warnings.filterwarnings("ignore")
from collections import namedtuple

random_sample_rate = 10 # random_sample_rate / 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_save = os.path.dirname(os.path.abspath(__file__)) + "/model/IL/Basic"
if not os.path.exists(model_save):
    os.makedirs(model_save)

para_save = os.path.dirname(os.path.abspath(__file__)) + "/para"
if not os.path.exists(para_save):
    os.makedirs(para_save)


parser = argparse.ArgumentParser(description="Training IL Agent")
parser.add_argument('-e', '--envs', nargs='+', help='Environment for training.', default=\
    ['RandomBayesGridEnv', 'BayesGridEnv', 'BasicGridEnv', 'ValidGridEnv', 'MazeGridEnv'])
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--demo_epoch', type=int, default=10)
parser.add_argument('--evaluate_epoch', type=int, default=50)
parser.add_argument('--pre_collect', type=int, default=5000)
parser.add_argument('--save_interval', type=int, default=20)
parser.add_argument('-l', '--log', default="")

def train(env, env_str, num_epochs=1000, batch_size=64, learning_rate=1e-3, demo_epoch=20, pre_collect=5000, save_interval=50, log_interval=20):
    model_state_dicts = {}
    model = ILNet().to(device)
    memory = ReplayBuffer()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bfs = BFSAgent()
    solvable_maze = 0
    bad_maze = 1

    maze_env = MazeGridEnv()

    # pre_collect
    while (len(memory) < pre_collect):
        obs = env.reset()
        history, solvable = bfs.solve(obs)
        if solvable:
            for h, a in history:
                memory.push(torch.tensor(h), torch.tensor(a))

    for epoch in range(num_epochs):        
        # get expert demo
        # for _ in range(demo_epoch):
        #     obs = env.reset()
        #     history, solvable = bfs.solve(obs)
        #     if solvable:
        #         solvable_maze += 1
        #         for h, a in history:
        #             memory.push(torch.tensor(h), torch.tensor(a))
        #     else:
        #         bad_maze += 1
        
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

        print ('\rEpoch [{}/{}], Loss: {:.4f}, Maze: {:.2f}, Score: {:.2f}'.format(epoch+1, num_epochs, loss.item(), solvable_maze / (solvable_maze + bad_maze), score), end="", flush=True)
        
        if (epoch+1) % log_interval == 0:
            print("")
        
        if (epoch+1) % save_interval == 0:
            model_state_dicts[str(epoch+1)] = copy.deepcopy(model.state_dict())
            torch.save(model_state_dicts, model_save + "/" + env_str)
            

def bayes_train(env, env_str, num_epochs=1000, batch_size=64, learning_rate=1e-3, evaluate_epoch=150, demo_epoch=20, pre_collect=5000, save_interval=50, log_interval=20):
    model_state_dicts = {}
    # evaluate_epoch = DATA_BEFORE_UPDATE + RANDOM_DATA_BEFORE_UPDATE * RANDOM_PARA_BEFORE_UPDATE
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

    # pre_collect
    # print("Start pre-collection")
    # t_memory = ReplayBuffer(capacity=10000)
    # while (len(t_memory) < pre_collect):
    #     obs = maze_env.reset()
    #     history, solvable = bfs.solve(obs)
    #     if solvable:
    #         for h, a in history:
    #             t_memory.push(torch.tensor(h), torch.tensor(a))
    # data_loader = torch.utils.data.DataLoader(dataset=t_memory, batch_size=batch_size, shuffle=True)
    # for _ in range(10):
    #     for i, (states, actions) in enumerate(data_loader):
    #         states = states.to(device)
    #         actions = actions.to(device)
    #         output_action_batch = model(states.unsqueeze(1).float())
    #         loss = criterion(output_action_batch, actions)
    #         loss = torch.mean(loss)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    lam = 0.8
    for epoch in range(num_epochs):
        scores = []
        for _ in range(evaluate_epoch):
            is_random = False
            if (random.randint(1, 10) <= random_sample_rate):
                env.generator.set_z(env.generator.random())
                is_random = True
            else:
                env.generator.set_z(env.generator.sample())
            # print(_, env.generator.get_z())

            pre_score, pre_solve, history, _bad_maze, _solvable_maze = evaluate(env, model, episodes=5, history_grid=history_grid, need_expert_demo=True)

            # print(_bad_maze, _solvable_maze, len(history))

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
                
                # for para, l in para_loss:
                #     env.generator.update(para, l)

                loss = torch.mean(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # env.generator.memory.calculate_mean()
            # env.generator.memory.save_status()
            # with torch.no_grad():
            #     for i, (paras, states, actions) in enumerate(data_loader):
            #         states = states.to(device)
            #         actions = actions.to(device)
            #         output_action_batch = model(states.float())
            #         loss = criterion(output_action_batch, actions)

            #         _loss = loss.clone()
            #         para_loss = zip(paras.cpu().detach().numpy(), _loss.cpu().detach().numpy())
                    
            #         for para, l in para_loss:
            #             env.generator.update(para, l)
            #         loss = torch.mean(loss)
            
        # evaluate
        after_score, after_solve, _history, _, _ = evaluate(maze_env, model, need_expert_demo=True, demo_every_reset=-1)

            # t_memory = ReplayBuffer(capacity=10000)
            # for h, a in _history:
            #     t_memory.push(torch.tensor(h), torch.tensor(a))
            # t_data_loader = torch.utils.data.DataLoader(dataset=t_memory, batch_size=batch_size, shuffle=False)
            # with torch.no_grad():
            #     for i, (states, actions) in enumerate(t_data_loader):
            #         states = states.to(device)
            #         actions = actions.to(device)
            #         output_action_batch = model(states.unsqueeze(1).float())
            #         t_loss = criterion(output_action_batch, actions).cpu().detach().numpy()
            #         for l in t_loss:
            #             env.generator.update(env.generator.get_z(), l)

        # lam *= (1+1e-3)
        # lam = np.clip(lam, 0, 0.9)
        # print(avg_score, pre_score, "=", lam*avg_score + (1-lam)*pre_score)

        # nei = np.mean(env.generator.query_nn(k=1))
        # scores.append((pre_solve, after_solve, pre_solve))
        # env.generator.update_LP(lam*avg_solve + (1-lam)*pre_score)
        # env.generator.update_ALP(lam*avg_solve + (1-lam)*pre_score - nei)
        # env.generator.update_ALP(lam*5*(after_target_solve-pre_target_solve) + (1-lam)*pre_solve)
        # env.generator.update_ALP(after_target_solve-pre_target_solve)

        print ('\rEpoch [{}/{}], Loss: {:.4f}, Maze: {:.2f}, Solve: {:.2f}, Lam: {:.2f}, Memories: {}'.format(epoch+1, num_epochs, loss.item(), solvable_maze / (solvable_maze + bad_maze + 1), after_solve, lam, len(memory)), end="", flush=True)
            # nei_score = env.generator.query_nn(k=1)
            # env.generator.update_LP(pre_score)
            # print(pre_score - np.mean(nei_score), nei_score)
            # env.generator.update_ALP(pre_score - np.mean(nei_score))
            # env.generator.update_ALP(pre_score)
        
        # env.generator.update_gp()

        if (epoch+1) % log_interval == 0:
            print ("")

        if (epoch+1) % save_interval == 0:
            model_state_dicts[str(epoch+1)] = copy.deepcopy(model.state_dict())
            with open(para_save + "/history" + env_str + ".json", "w+") as f:
                json.dump(history_grid, f)
            torch.save(model_state_dicts, model_save + "/" + env_str)

def evaluate(env, model, history_grid=None, episodes=50, demo_every_reset=1,need_expert_demo=False):
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
                # print(history_grid)
                while True:
                    action = model.predict(obs)
                    obs, reward, done, info = env.step(action)
                    if done:
                        if info.get('success'):
                            values.append(0.2)
                            solve.append(1)
                        else:
                            if added < demo_every_reset or demo_every_reset == -1:
                                added += 1
                                history.extend(h)
                                if history_grid != None and added == 1:
                                    history_grid.append(obs.tolist())
                            values.append(1)
                            solve.append(0)
                        break
            else:
                bad_maze += 1
                values.append(-1)
        else:
            # maze
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
            bayes_train(env, env_str=args.envs[i]+args.log, num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, evaluate_epoch=args.evaluate_epoch ,demo_epoch=args.demo_epoch, pre_collect=args.pre_collect, save_interval=args.save_interval)
        else:
            train(env, env_str=args.envs[i], num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, demo_epoch=args.demo_epoch, pre_collect=args.pre_collect, save_interval=args.save_interval)