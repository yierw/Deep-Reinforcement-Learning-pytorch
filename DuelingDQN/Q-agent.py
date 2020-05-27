import gym
import random
import torch
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.999             # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 5        # how often to update the target network

class Agent():
    def __init__(self, state_size, action_size, hidden_size, seed):

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.seed = seed

        self.online_net = QNetwork(state_size, action_size, hidden_size, seed).to(device)
        self.target_net = QNetwork(state_size, action_size, hidden_size, seed).to(device)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr = LR)
        self.memory = ReplayBuffer(buffer_size = BUFFER_SIZE)
        self.t_step = 1 # tracking whether to update target network parameters

    def act(self, state, eps = 0.):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # select action according to online network
        self.online_net.eval()
        with torch.no_grad():
            action = self.online_net(state_tensor).argmax(1).item()
        self.online_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return action
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        loss_fn = nn.MSELoss()

        next_Q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target = rewards + gamma*next_Q*(1-dones)

        prediction = self.online_net(states).gather(1, actions)

        loss = loss_fn(prediction, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, tau):
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(tau*target_param.data + (1.0-tau)*online_param.data)


    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = self.t_step + 1

        # update target network
        if (self.t_step % UPDATE_EVERY) == 0:
            self.soft_update(TAU)

        # sample batch and learn
        if len(self.memory)> BATCH_SIZE:
            experiences = self.memory.sample(BATCH_SIZE)
            self.learn(experiences, GAMMA)


class ReplayBuffer(object):
    def __init__(self, buffer_size) :
        self.memory = deque(maxlen = buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        x = self.experience(state, action, reward, next_state, done)
        self.memory.append(x)

    def sample(self, batch_size):
        samples = random.sample(self.memory, k = batch_size)
        batch = self.experience(*zip(*samples))
        states = torch.from_numpy(np.asarray(batch.state)).float().to(device)
        actions = torch.from_numpy(np.asarray(batch.action)).long().view(-1,1).to(device) # discrete action space
        rewards = torch.from_numpy(np.asarray(batch.reward)).float().view(-1,1).to(device)
        next_states = torch.tensor(np.asarray(batch.next_state)).float().to(device)
        # 0 for note finished, 1 for terminated
        dones = torch.tensor([1 if done else 0 for done in batch.done]).float().view(-1,1).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.main = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        x = self.main(state)
        return x
