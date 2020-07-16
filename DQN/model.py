import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, o_dim, a_dim, h_dim = 64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(o_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, a_dim)
        )

    def forward(self, state):
        return self.main(state)


class ConvBase(nn.Module):
    """
    used for Atari games
    """
    def __init__(self, c_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(c_dim, 16, kernel_size = 8, stride = 4, bias = False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)
        #(:, 32, 8, 8)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        return  x.view(-1, 2048) 

class ConvQNet(nn.Module):
    def __init__(self, c_dim, a_dim):
        super().__init__()
        self.features = ConvBase(c_dim)
        self.fc_layer = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, a_dim)
        )

    def forward(self, state):
        x = self.features(state)
        return self.fc_layer(x)
