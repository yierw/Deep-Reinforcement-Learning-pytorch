import torch
import torch.nn as nn

class QNetwork(nn.Module):

    hidden_dim = 64

    def __init__(self, input_dim, output_dim, seed = 1234):
        super(QNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.seed = torch.manual_seed(seed)

        self.main = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, state):
        x = self.main(state)
        return x
