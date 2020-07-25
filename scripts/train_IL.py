import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mygrid.MiniGrid.Generator.HyperPara import DATA_BEFORE_UPDATE, RANDOM_DATA_BEFORE_UPDATE, RANDOM_PARA_BEFORE_UPDATE, GRID_HEIGHT, GRID_WIDTH
from mygrid.MiniGrid.Utils.BFS import BFSAgent
from mygrid.MiniGrid.BayesGridEnv import BayesGridEnv
from mygrid.MiniGrid.Agent import ILNet
from mygrid.MiniGrid.Memory import ReplayBuffer
import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save = os.path.dirname(os.path.abspath(__file__)) + "/model"
if not os.path.exists(model_save):
    os.mkdir(model_save)


num_epochs = 100
batch_size = 64
learning_rate = 0.001

# evaluate_epoch = DATA_BEFORE_UPDATE + RANDOM_DATA_BEFORE_UPDATE * RANDOM_PARA_BEFORE_UPDATE
evaluate_epoch = 10
demo_epoch = 20
update_iter = 50

model = ILNet().to(device)
env = BayesGridEnv()
memory = ReplayBuffer()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

bfs = BFSAgent()

count = 0
solvable_maze = 0
bad_maze = 0
for epoch in range(num_epochs):
    # evaluation
    for _ in range(evaluate_epoch):
        obs = env.reset()
        while True:
            obs, reward, done, info = env.step(model.predict(obs))
            if done:
                break
    
    # get expert demo
    for _ in range(demo_epoch):
        while True:
            obs = env.reset()
            history, solvable = bfs.solve(obs)
            if solvable:
                count += len(history)

                solvable_maze += 1

                for h, a in history:
                    memory.push(torch.tensor(h), torch.tensor([a]))
            else:
                bad_maze += 1

            if (count > 2*batch_size):
                break
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

    if (epoch+1) % 1 == 0:
        print ('\rEpoch [{}/{}], Loss: {:.4f}, Maze: {:.2f}'.format(epoch+1, num_epochs, loss.item(), solvable_maze / (solvable_maze + bad_maze)), end="", flush=True)

    if (epoch+1) % 20 == 0:
        torch.save(model.state_dict(), model_save + "/IL_{}.pkl".format(epoch+1))


