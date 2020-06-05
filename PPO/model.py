import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.main= nn.Sequential(
            # (:,2,80,80) to (:,4,38,38)
            nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False),
            nn.ReLU(inplace = True),
            # (:,4,38,38) to (:,16,9,9)
            nn.Conv2d(4, 16, kernel_size=6, stride=4),
            nn.ReLU(inplace = True),
            nn.Flatten(),

            nn.Linear(9*9*16, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 1),

            nn.Sigmoid()
        )

    def forward(self, states):
        return self.main(states)
