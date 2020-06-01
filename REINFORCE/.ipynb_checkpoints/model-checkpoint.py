import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(Policy, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_size, action_size),
            nn.ReLU(inplace = True),
            nn.Softmax(dim=1) 
        )

    def forward(self, state):
        # output prob dist over action space
        x = self.main(state)
        return x
    
    def act(self, state):
        # batch size is 1
        state = torch.from_numpy(state).float().view(1,-1).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        # sample one action from prob distribution
        action = m.sample() 
        # out put the sampled action and the log prob
        return action.item(), m.log_prob(action)
