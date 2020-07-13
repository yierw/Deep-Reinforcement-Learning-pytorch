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


class ConvBase(nn.Module):
    """
    used for Atari games
    """
    def __init__(self, c_dim, a_dim, seed = 1234):
        super().__init__()

        torch.manual_seed(seed)

        self.main = nn.Sequential(
            nn.Conv2d(c_dim, 32, kernel_size=8, stride = 4, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dim)
        )

    def forward(self, state):
        return  self.main(state)
