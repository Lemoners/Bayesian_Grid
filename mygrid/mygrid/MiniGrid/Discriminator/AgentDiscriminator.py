from ..Utils.BFS import BFSAgent
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128

def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1/(1 + np.exp(-x))


class AgentDiscriminator(object):
    def __init__(self):
        super().__init__()
        self.bfs = BFSAgent()

    def evaluate_agent(self, model, grids, threshold=0.5):
        """
        When setting a threshold, the agent will only be told that: (Just like human)
            1. You're doing Good.
            2. You're doing Bad.
        """
        model.eval()
        model = model.to(device)
        
        history = []
        for grid in grids:
            h, s = self.bfs.solve(grid)
            if s:
                history.extend(h)        
        data_loader = torch.utils.data.DataLoader(dataset=history, batch_size=batch_size, shuffle=False)
        
        precision = 0

        for i, (states, actions) in enumerate(data_loader):
            states = states.to(device).unsqueeze(1).float()
            actions = actions.to(device)
            predictions = model(states)
            predictions = torch.argmax(predictions, dim=1)
            predictions = (predictions - actions).view(actions.size(0))
            predictions = (predictions == 0).sum(dim=0)
            precision += predictions.item()
        model.train()
        precision = precision / len(history)
        return (sigmoid(40 * max(precision - 0.9, 0) - 2))
        # return ((precision / len(history)) > threshold)









