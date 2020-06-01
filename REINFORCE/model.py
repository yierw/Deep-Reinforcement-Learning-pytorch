import torch
import torch.nn as nn

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
