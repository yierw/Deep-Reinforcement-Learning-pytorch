import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from buffer import ReplayBuffer

def soft_update(target, source, tau):
    """ Perform soft update"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class DQNAgent:
    """
    DQN Agent, valid for discrete actioin space
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()
    iter = 0
    
    def __init__(self, net, o_dim, a_dim, lr = 1e-3, batch_size = 16, algorithm = "ddqn",
                 gamma = 0.99, tau = 1e-3, buffer_size = int(1e6)):
        """
        o_dim: observation space dim (or # of channels)
        a_dim: action space dimension
        """
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size

        if algorithm.lower() in ("dqn"):
            self.algorithm = "dqn"
        elif algorithm.lower() in ("ddqn", "double dqn", "doubledqn"):
            self.algorithm = "ddqn"
        else:
            raise TypeError("cannot recognize algorithm")

        self.buffer = ReplayBuffer(buffer_size, batch_size)

        self.online_net = net(o_dim , a_dim).to(self.device)
        self.target_net = net(o_dim , a_dim).to(self.device)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr = lr)

    def get_action(self, state, eps = 0.):
        """ Epsilon-greedy action selection """
        
        if random.random() > eps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            self.online_net.eval()
            with torch.no_grad():
                action = self.online_net(state_tensor).argmax(1).item()
            self.online_net.train()
            
            return action
        else:
            return random.choice(np.arange(self.a_dim))

    def update(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        actions = torch.LongTensor(actions).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        dones = torch.FloatTensor(dones).view(-1, 1).to(self.device)
        
        if self.algorithm == "ddqn":
            max_actions = self.online_net(next_states).max(1)[1].view(-1, 1)
            Q_next = self.target_net(next_states).gather(1, max_actions)

        elif self.algorithm == "dqn":
            Q_next = self.target_net(next_states).max(1)[0].view(-1, 1)
        else:
            raise TypeError("cannot recognize algorithm")

        Q_targets = rewards + self.gamma * Q_next * (1. -dones)
        Q_expected = self.online_net(states).gather(1, actions)
        
        loss = self.loss_fn(Q_expected, Q_targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.)
        self.optimizer.step()

    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        if len(self.buffer) > self.batch_size:
            experiences = self.buffer.sample()
            self.update(experiences)
            soft_update(self.target_net, self.online_net, self.tau)
            self.iter += 1