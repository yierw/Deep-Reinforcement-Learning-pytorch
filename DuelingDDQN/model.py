import torch
import torch.nn as nn

class duelingNet(nn.Module):
    def __init__(self, action_dim):
        super(duelingNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6912, 512),
            nn.ReLU()
        )

        self.value_layer = nn.Linear(512,1)
        self.adv_layer = nn.Linear(512,action_dim)


    def forward(self, state):
        x = self.features(state)
        v,a = self.value_layer(x), self.adv_layer(x)
        return  v - a + a.mean(dim=1).unsqueeze(1)
