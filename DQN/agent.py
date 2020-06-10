### implement prioritized experience replay
import sys
import random
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.append('../')
from common.replaybuffer import *
from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001             # for soft update of target parameters
LR = 1e-3               # learning rate
UPDATE_EVERY = 1        # how often to update the target network
LEARN_NUM = 1

class PERAgent():
    def __init__(self, state_size, action_size, algorithm = "dqn"):
        if algorithm.lower() in ("dqn"):
            self.algorithm = "dqn"
        elif algorithm.lower() in ("ddqn", "double dqn", "doubledqn"):
            self.algorithm = "ddqn"
        else:
            raise TypeError("cannot recognize algorithm")
        
        self.state_size = state_size
        self.action_size = action_size

        self.online_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr = LR)
        self.buffer = PrioritizedBuffer(BUFFER_SIZE)
        self.t_step = 0 # tracking whether to update target network parameters

    def get_action(self, state, eps = 0.):
        # Epsilon-greedy action selection
        if random.random() > eps:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # select action according to online network
            self.online_net.eval()
            with torch.no_grad():
                action = self.online_net(state_tensor).argmax(1).item()
            self.online_net.train()
            return action
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        idxs, IS_weights, states, actions, rewards, next_states, dones = experiences
        IS_weights = torch.FloatTensor(IS_weights).to(device)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        if self.algorithm == "ddqn":
            next_actions = self.online_net(next_states).max(1)[1].unsqueeze(1)
            next_Q = self.target_net(next_states).gather(1, next_actions)

        elif self.algorithm == "dqn":
            next_Q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        else:
            raise TypeError("cannot recognize algorithm")
                      
        target = rewards + gamma * next_Q * (1-dones)
        prediction = self.online_net(states).gather(1, actions)
        TD_errors = torch.abs(target.detach() - prediction).squeeze()
        # update online network
        loss = torch.mean(torch.pow(TD_errors, 2)* IS_weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update priorities
        self.buffer.update_priority(idxs, TD_errors.cpu().detach().numpy())
        
    def soft_update(self, model, target_model, tau):
        """
        tau = 1.0 --> hard copy 
        tau < 1.0 --> soft update target network parameters
        """
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_((1.0-tau)*target_param.data + tau*param.data)

    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        if len(self.buffer) > BATCH_SIZE:
            self.t_step = self.t_step + 1
            for _ in range(LEARN_NUM):
                experiences = self.buffer.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)
            if (self.t_step % UPDATE_EVERY) == 0:
                self.soft_update(self.online_net, self.target_net, TAU)
                
                
                
class Agent():
    def __init__(self, state_size, action_size, algorithm = "dqn"):
        if algorithm.lower() in ("dqn"):
            self.algorithm = "dqn"
        elif algorithm.lower() in ("ddqn", "double dqn", "doubledqn"):
            self.algorithm = "ddqn"
        else:
            raise TypeError("cannot recognize algorithm")

        self.state_size = state_size
        self.action_size = action_size

        self.online_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr = LR)
        self.buffer = ReplayBuffer(buffer_size = BUFFER_SIZE)
        self.t_step = 0 # tracking whether to update target network parameters

    def get_action(self, state, eps = 0.):
        # Epsilon-greedy action selection
        if random.random() > eps:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # select action according to online network
            self.online_net.eval()
            with torch.no_grad():
                action = self.online_net(state_tensor).argmax(1).item()
            self.online_net.train()
            return action
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        if self.algorithm == "ddqn":
            next_actions = self.online_net(next_states).max(1)[1].unsqueeze(1)
            next_Q = self.target_net(next_states).gather(1, next_actions)

        elif self.algorithm == "dqn":
            next_Q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        else:
            raise TypeError("cannot recognize algorithm")

        target = rewards + gamma * next_Q * (1-dones)
        prediction = self.online_net(states).gather(1, actions)

        loss_fn = nn.MSELoss()
        loss = loss_fn(prediction, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, model, target_model, tau):
        """
        tau = 1.0 --> hard copy 
        tau < 1.0 --> soft update target network parameters
        """
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_((1.0-tau)*target_param.data + tau*param.data)

    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        if len(self.buffer)> BATCH_SIZE:
            self.t_step = self.t_step + 1
            for _ in range(LEARN_NUM):
                experiences = self.buffer.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)
            if (self.t_step % UPDATE_EVERY) == 0:
                self.soft_update(self.online_net, self.target_net, TAU)
                
                