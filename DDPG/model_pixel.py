import torch
import torch.nn as nn

class CentralizedCritic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(CentralizedCritic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.in_layer = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            nn.Linear(64 + self.action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, a):
        x = self.in_layer(x)
        xa = torch.cat((x,a), dim = 1)
        return self.main(xa)

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        return self.main(obs)
