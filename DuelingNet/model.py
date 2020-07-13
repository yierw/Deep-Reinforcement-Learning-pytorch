import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBase(nn.Module):
    """
    used for Atari games
    """
    def __init__(self, c_dim, seed):
        super().__init__()

        torch.manual_seed(seed)
        
        #self.conv1 = nn.Conv2d(c_dim, 32, kernel_size=8, stride = 4, bias = False)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        #self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        #(:, 64, 6, 6)
        
        self.conv1 = nn.Conv2d(c_dim, 16, kernel_size = 8, stride = 4, bias = False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)
        #(:, 32, 8, 8)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        out = x.view(-1, 2048) 
        return  out

class DuelingNet(nn.Module):
    def __init__(self, c_dim, a_dim, seed = 1234):
        super().__init__()
        torch.manual_seed(seed)

        self.features = ConvBase(c_dim, seed)

        self.value_layer = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, a_dim)
        )

    def forward(self, state):
        x = self.features(state)
        v, a = self.value_layer(x), self.adv_layer(x)
        q = v - a + a.mean(dim=1).unsqueeze(1)
        return q