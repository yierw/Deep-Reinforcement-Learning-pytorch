import torch
import torch.nn as nn

class duelingNet(nn.Module):
    """
    used for Atari games
    """
    def __init__(self, channel_dim, action_dim, seed = 1234):
        super(duelingNet, self).__init__()

        self.channel_dim = channel_dim
        self.action_dim = action_dim
        self.seed = torch.manual_seed(seed)

        self.features = nn.Sequential(
            nn.Conv2d(self.channel_dim, 32, kernel_size=8, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.value_layer = nn.Sequential(
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512,1)
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dim)
        )


    def forward(self, state):
        x = self.features(state)
        v,a = self.value_layer(x), self.adv_layer(x)
        return  v - a + a.mean(dim=1).unsqueeze(1)


class ConvQNet(nn.Module):
    """
    used for Atari games
    """
    def __init__(self, channel_dim, action_dim, seed = 1234):
        super(ConvQNet, self).__init__()

        self.channel_dim = channel_dim
        self.action_dim = action_dim
        self.seed = torch.manual_seed(seed)

        self.main = nn.Sequential(
            nn.Conv2d(self.channel_dim, 32, kernel_size=8, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dim)
        )

    def forward(self, state):
        return  self.main(state)


class QNetwork(nn.Module):

    hidden_dim = 64

    def __init__(self, obs_dim, action_dim, seed = 1234):
        super(QNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.seed = torch.manual_seed(seed)

        self.main = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

    def forward(self, state):
        x = self.main(state)
        return self.main(state)
