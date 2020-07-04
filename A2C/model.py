import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoHeadNetwork(nn.Module):

    def __init__(self, o_dim, a_dim, seed = 1234, h_dim = 64):
        super(TwoHeadNetwork, self).__init__()

        self.o_dim = o_dim
        self.a_dim = a_dim
        torch.manual_seed(seed)

        self.features = nn.Sequential(
            nn.Linear(o_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        self.actor_fc = nn.Linear(h_dim, a_dim)

        self.critic_fc = nn.Linear(h_dim, 1)


    def forward(self, state):
        x = self.features(state)
        prob =  F.softmax(self.actor_fc(x), 1)
        value = self.critic_fc(x)
        return  prob, value
