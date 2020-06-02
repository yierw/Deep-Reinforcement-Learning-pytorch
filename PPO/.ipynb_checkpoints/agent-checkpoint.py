
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class policy_net(nn.Module):
    def __init__(self):
        super(policy_net, self).__init__()
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
        probs = self.main(states)
        return probs

class Agent():
    def __init__(self, learning_rate = 0.1):
        self.policy = policy_net().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = learning_rate)

    def act(self, states):
        probs = self.policy(states).squeeze().cpu().detach().numpy()
        actions = np.where(np.random.rand(states.size(0)) < probs, RIGHT, LEFT)
        probs = np.where(actions==RIGHT, probs, 1.0-probs)
        # output numpy array
        return actions, probs

    def get_probs(self, states, actions):
        probs = self.policy(states)
        actions = torch.tensor(actions, dtype=torch.int8, device=device).view(-1,1)
        p_action = torch.where(actions == RIGHT, probs, 1.0-probs)
        # output tensor (includes grad)
        return p_action
