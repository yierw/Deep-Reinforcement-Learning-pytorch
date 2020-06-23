import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from buffer import ReplayBuffer

from model import Critic, Actor
from OUNoise import OUNoise

def soft_update(target, source, tau):
    """ Perform soft update"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DDPGAgent():
    """
    DQPG Agent, valid for continuous actioin space
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()
    iter = 0
    t_step = 0

    def __init__(self, o_dim, a_dim, lr_actor = 1e-4, lr_critic = 1e-3, weight_decay = 1e-2,
                 batch_size = 64, gamma = 0.99, tau = 0.001, buffer_size = int(1e6),
                 update_every = 10, seed = 1234):

        """
        o_dim/c_dim: observation space dimension/ # of channels when image as input
        a_dim: action space dimension
        """

        self.o_dim = o_dim
        self.a_dim = a_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.update_every = update_every
        self.seed = seed

        self.buffer = ReplayBuffer(self.buffer_size, self.batch_size, self.seed)

        # Actor Network (w/ Target Network)
        self.actor = Actor(self.o_dim , self.a_dim, self.seed).to(self.device)
        self.target_actor = Actor(self.o_dim , self.a_dim, self.seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = self.lr_actor)
        # Critic Network (w/ Target Network)
        self.critic = Critic(self.o_dim , self.a_dim, self.seed).to(self.device)
        self.target_critic = Critic(self.o_dim , self.a_dim, self.seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = self.lr_critic, weight_decay = self.weight_decay)
        # Noise process
        self.noise = OUNoise(self.a_dim)

    def get_action(self, state, add_noise = True):
        state_tensor = torch.FloatTensor(states).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor)[0].detach().cpu().numpy()
        self.actor.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def get_action2(self, state, eps = 0.):
        """
        for multibinary action space
        note: although the action space is multi-binary, float vectors are fine
        for slimevolly gym environment

        forward = True if action[0]>0 else False
        backward = True if action[1]>0 elseTrue False
        jump = True if action[2]>0 else True False

        """
        if random.random() > eps:
            # select action according to online network
            state_tensor = torch.FloatTensor(states).unsqueeze(0).to(self.device)
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state_tensor)[0].detach().cpu().numpy()
            self.actor.train()
            return action

        else:
            return random.choices([0,1], k = self.a_dim)

    def update(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        actions = torch.FloatTensor(actions).to(self.device)

        rewards = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        dones = torch.FloatTensor(dones).view(-1, 1).to(self.device)

        # ---------------------------- update critic ---------------------------- #
        next_actions = self.target_actor(next_states)
        Q_next = self.target_critic(next_states, actions_next)
        Q_targets = rewards + self.gamma * Q_next * (1. -dones)
        Q_expected = self.critic(states, actions)

        loss = self.loss_fn(Q_expected, Q_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.)
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
        self.actor_optimizer.step()
        # ---------------------------- update target net ---------------------------- #
        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_actor, self.actor, self.tau)

    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        if (len(self.buffer) <= self.batch_size):
            pass
        else:
            """
            first update happens when we have enough tuples for a batch;
            then update after push update_every tuples
            """
            if self.t_step % self.update_every == 0:
                experiences = self.buffer.sample()
                self.update(experiences)
                self.iter += 1

            self.t_step += 1

    def reset(self):
        self.noise.reset()
