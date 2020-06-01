import random
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim

from model import QNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3             # for soft update of target parameters
LR = 5e-4               # learning rate

class Agent():
    def __init__(self, action_dim, channel_dim):
        self.action_dim = action_dim

        self.online_net = QNetwork(action_dim, channel_dim).to(device)
        self.target_net = QNetwork(action_dim, channel_dim).to(device)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr = LR)
        self.buffer = ReplayBuffer(buffer_size = BUFFER_SIZE)

        self.t_step = 0 #  track learning
        self.loss = 999.

    def act(self, state, eps = 0.):
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
            return random.choice(np.arange(self.action_dim))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        # double DQN
        next_actions = self.target_net(next_states).max(1)[1].unsqueeze(1)
        next_Q = self.target_net(next_states).gather(1, next_actions)
        target = rewards + gamma * next_Q * (1-dones)
        prediction = self.online_net(states).gather(1, actions)
        # compute loss
        loss_fn = nn.MSELoss()
        loss = loss_fn(prediction, target.detach())
        self.t_step = self.t_step + 1
        self.loss = loss.item()
        # update critic
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.online_net.parameters(), 10.)
        self.optimizer.step()

    def soft_update(self, model, target_model, tau):
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)

    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        if len(self.buffer)> WARM_UP:
            # sample a minibatch
            experiences = self.buffer.sample(BATCH_SIZE)
            # update critic
            self.learn(experiences, GAMMA)
            # soft upfate target networks
            self.soft_update(self.online_net, self.target_net, TAU)


class ReplayBuffer(object):
    def __init__(self, buffer_size, num_agents = 1) :
        self.memory = deque(maxlen = buffer_size)
        self.num_agents = num_agents
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # 0 for note finished (False), 1 for terminated (True)
        self.map_done = lambda x: 1 if x else 0

    def push(self, state, action, reward, next_state, done):
        done = self.map_done(done)
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.memory, k = batch_size)
        batch = self.experience(*zip(*samples))

        states = torch.from_numpy(np.asarray(batch.state)).float().to(device)
        next_states = torch.tensor(np.asarray(batch.next_state)).float().to(device)
        # discrete action space
        actions = torch.from_numpy(np.asarray(batch.action)).long().view(-1,1).to(device)
        rewards = torch.from_numpy(np.asarray(batch.reward)).float().view(-1,1).to(device)
        dones = torch.from_numpy(np.asarray(batch.done)).float().view(-1,1).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

def preprocess_frames(frame_list, bkg_color = np.array([144, 72, 17])):
    x = np.asarray(frame_list)[:,52:152,30:130,:]-bkg_color
    return np.mean(x, axis=-1)/255.

def collect_tuple(env, action, k=4):
    frame_list = []
    reward_list = []
    for _ in range(k):
        frame, r, done, _ = env.step(action)
        frame_list.append(frame)
        reward_list.append(r)

    reward = np.sum(reward_list)
    next_state = preprocess_frames(frame_list)
    return reward, next_state, done
