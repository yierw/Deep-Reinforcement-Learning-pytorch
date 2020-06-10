import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 64):
        super(QNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state):
        x = self.main(state)
        return x
