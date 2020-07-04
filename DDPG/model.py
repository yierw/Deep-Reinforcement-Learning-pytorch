import numpy as np
import torch
import torch.nn as nn

def weights_init_uniform_rule(layer):
    classname = layer.__class__.__name__
    if classname.find('Linear') != -1:
        y = 1.0/np.sqrt(layer.in_features)
        layer.weight.data.uniform_(-y, y)
        layer.bias.data.fill_(0.0)

class Actor(nn.Module):
    def __init__(self, o_dim, a_dim, h_dim, reset = False, seed = 1234):
        super(Actor, self).__init__()

        self.o_dim = o_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.reset = reset
        self.seed = torch.manual_seed(seed)

        self.main = nn.Sequential(
            nn.Linear(self.o_dim, self.h_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.h_dim, self.a_dim),
            nn.Tanh()
        )

        if self.reset:
            self.reset_parameters()

    def reset_parameters(self):
        self.main.apply(weights_init_uniform_rule)

    def forward(self, obs):
        return self.main(obs)


class Critic(nn.Module):
    def __init__(self, o_dim, a_dim, h_dim, reset = False, seed = 1234):
        super(Critic, self).__init__()

        self.o_dim = o_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.reset = reset
        self.seed = torch.manual_seed(seed)

        self.obs_fc = nn.Sequential(
            nn.Linear(self.o_dim, self.h_dim),
            nn.ReLU(inplace = True),
        )

        self.main = nn.Sequential(
            nn.Linear(self.h_dim + self.a_dim, self.h_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.h_dim, 1)
        )

        if self.reset:
            self.reset_parameters()

    def reset_parameters(self):
        self.main.apply(weights_init_uniform_rule)
        self.obs_fc.apply(weights_init_uniform_rule)

    def forward(self, x, a):
        x = self.obs_fc(x)
        xa = torch.cat((x,a), dim = 1)
        return self.main(xa)
