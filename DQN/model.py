import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, obs_dim, a_dim, h_dim = 64, seed = 1234):
        super().__init__()

        torch.manual_seed(seed)

        self.main = nn.Sequential(
            nn.Linear(o_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, a_dim)
        )

    def forward(self, state):
        x = self.main(state)
        return self.main(state)


class ConvBase(nn.Module):

    def __init__(self, c_dim, seed):
        super().__init__()

        torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(c_dim, 32, kernel_size=8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        out = x.view(-1, 2304) # 64 * 6 * 6
        return  out

class ConvQNet(nn.Module):
    def __init__(self, c_dim, a_dim, seed = 1234):
        super().__init__()

        torch.manual_seed(seed)

        self.features = ConvBase(c_dim, seed)

        self.fc_layer = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, a_dim)
        )

    def forward(self, state):
        x = self.features(state)
        q = self.fc_layer(x)
        return q
