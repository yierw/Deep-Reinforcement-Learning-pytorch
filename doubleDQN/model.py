import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, action_dim, channel_dim = 4):
        super(QNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channel_dim, 32, kernel_size=8, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5184, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, state):
        return  self.main(state)
