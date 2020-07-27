import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mygrid.MiniGrid.Generator.HyperPara import DATA_BEFORE_UPDATE, RANDOM_DATA_BEFORE_UPDATE, RANDOM_PARA_BEFORE_UPDATE, GRID_HEIGHT, GRID_WIDTH
from mygrid.MiniGrid.Utils.BFS import BFSAgent
from mygrid.MiniGrid.BayesGridEnv import BayesGridEnv
from mygrid.MiniGrid.Agent import ILNet
from mygrid.MiniGrid.Memory import ReplayBuffer
from mygrid.MiniGrid import BasicGridEnv, ValidGridEnv, MazeGridEnv
from mygrid.MiniGrid.HardGridEnv import HardGridEnv
from mygrid.MiniGrid.Generator.HardMazeGene import HardMazeGene
from mygrid.MiniGrid.Discriminator.AgentDiscriminator import AgentDiscriminator
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
# import warnings
# warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_save = os.path.dirname(os.path.abspath(__file__)) + "/model/IL/Basic"
if not os.path.exists(model_save):
    os.makedirs(model_save)

para_save = os.path.dirname(os.path.abspath(__file__)) + "/para"
if not os.path.exists(para_save):
    os.makedirs(para_save)


parser = argparse.ArgumentParser(description="Training IL Agent")
parser.add_argument('-e', '--envs', nargs='+', help='Environment for training.', default=\
    ['BayesGridEnv', 'BasicGridEnv', 'ValidGridEnv', 'MazeGridEnv'])
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--demo_epoch', type=int, default=20)
parser.add_argument('--pre_collect', type=int, default=5000)
parser.add_argument('--save_interval', type=int, default=20)


def train(env, env_str, num_epochs=1000, batch_size=64, learning_rate=1e-3, demo_epoch=20, pre_collect=5000, save_interval=50, log_interval=50):
    model_state_dicts = {}
    model = ILNet().to(device)
    memory = ReplayBuffer()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bfs = BFSAgent()
    solvable_maze = 0
    bad_maze = 0

    # pre_collect
    while (len(memory) < pre_collect):
        obs = env.reset()
        history, solvable = bfs.solve(obs)
        if solvable:
            for h, a in history:
                memory.push(torch.tensor(h), torch.tensor([a]))

    for epoch in range(num_epochs):        
        # get expert demo
        for _ in range(demo_epoch):
            obs = env.reset()
            history, solvable = bfs.solve(obs)
            if solvable:
                solvable_maze += 1
                for h, a in history:
                    memory.push(torch.tensor(h), torch.tensor([a]))
            else:
                bad_maze += 1

        # optimize
        transition = memory.sample(batch_size)
        batch = memory.memtype(*zip(*transition))
        state_batch = torch.cat(batch.state).view(-1, 1, GRID_HEIGHT, GRID_WIDTH).float().to(device)
        action_batch = torch.cat(batch.action).to(device)
        output_action_batch = model(state_batch).float().to(device)
        
        loss = criterion(output_action_batch, action_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print ('\rEpoch [{}/{}], Loss: {:.4f}, Maze: {:.2f}'.format(epoch+1, num_epochs, loss.item(), solvable_maze / (solvable_maze + bad_maze)), end="", flush=True)
        
        if (epoch+1) % log_interval == 0:
            print("")
        

        if (epoch+1) % save_interval == 0:
            model_state_dicts[str(epoch+1)] = model.state_dict().copy()
    print("")
    torch.save(model_state_dicts, model_save + "/" + env_str)
            

def bayes_train(env, env_str, num_epochs=1000, batch_size=64, learning_rate=1e-3, evaluate_epoch=10, demo_epoch=20, pre_collect=5000, save_interval=50, log_interval=50):
    model_state_dicts = {}
    # evaluate_epoch = DATA_BEFORE_UPDATE + RANDOM_DATA_BEFORE_UPDATE * RANDOM_PARA_BEFORE_UPDATE
    model = ILNet().to(device)
    env = BayesGridEnv()
    memory = ReplayBuffer()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bfs = BFSAgent()

    solvable_maze = 0
    bad_maze = 0

    agent_dis = AgentDiscriminator()
    hard_maze_gene = HardMazeGene()

    # pre_collect
    while (len(memory) < pre_collect):
        obs = env.reset()
        history, solvable = bfs.solve(obs)
        if solvable:
            for h, a in history:
                memory.push(torch.tensor(h), torch.tensor([a]))

    for epoch in range(num_epochs):
        # evaluation
        mazes = hard_maze_gene.batch_gene(batches=evaluate_epoch)
        score = agent_dis.evaluate_agent(model, mazes)
        env._update_model(score)
        
        for _ in range(evaluate_epoch):
            obs = env.reset()
            while True:
                action = model.predict(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    break
        
        # get expert demo
        for _ in range(demo_epoch):
            obs = env.reset()
            history, solvable = bfs.solve(obs)
            if solvable:
                solvable_maze += 1
                for h, a in history:
                    memory.push(torch.tensor(h), torch.tensor([a]))
            else:
                bad_maze += 1

        # optimize
        transition = memory.sample(batch_size)
        batch = memory.memtype(*zip(*transition))
        state_batch = torch.cat(batch.state).view(-1, 1, GRID_HEIGHT, GRID_WIDTH).float().to(device)
        action_batch = torch.cat(batch.action).to(device)
        output_action_batch = model(state_batch).float().to(device)
        
        loss = criterion(output_action_batch, action_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print ('\rEpoch [{}/{}], Loss: {:.4f}, Maze: {:.2f}'.format(epoch+1, num_epochs, loss.item(), solvable_maze / (solvable_maze + bad_maze)), end="", flush=True)
        if (epoch+1) % log_interval == 0:
            print ("")

        if (epoch+1) % save_interval == 0:
            model_state_dicts[str(epoch+1)] = model.state_dict().copy()
    
    with open(para_save + "/history.json", "w+") as f:
        json.dump(env.generator.history_z, f)

    torch.save(model_state_dicts, model_save + "/" + env_str)



if __name__ == "__main__":
    args = parser.parse_args()
    envs = [eval(e) for e in args.envs]
    for i, _env in enumerate(envs):
        env = _env()
        print("Training IL Agent in {}".format(args.envs[i]))
        if 'Bayes' in args.envs[i]:
            bayes_train(env, env_str=args.envs[i], num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, demo_epoch=args.demo_epoch, pre_collect=args.pre_collect, save_interval=args.save_interval)
        else:
            train(env, env_str=args.envs[i], num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, demo_epoch=args.demo_epoch, pre_collect=args.pre_collect, save_interval=args.save_interval)