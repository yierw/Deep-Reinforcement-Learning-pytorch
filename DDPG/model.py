import torch
import torch.nn as nn

class Critic(nn.Module):

    hidden_dim = 128

    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.in_layer = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x, a):
        x = self.in_layer(x)
        xa = torch.cat((x,a), dim = 1)
        return self.main(xa)

class Actor(nn.Module):

    hidden_dim = 128

    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        return self.main(obs)
