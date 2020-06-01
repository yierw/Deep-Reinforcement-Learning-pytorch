import random
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from model import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LR = 1e-2

class Agent():
    def __init__(self, state_size, hidden_size, action_size):
        self.policy = Policy(state_size, hidden_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = LR)

    def act(self, state):
        state = torch.from_numpy(state).float().view(1,-1).to(device)
        probs = self.policy(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        log_prob =  m.log_prob(action)
        return action.item(), log_prob
