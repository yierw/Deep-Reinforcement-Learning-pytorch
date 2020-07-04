import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from buffer import ReplayBuffer
from OUNoise import OUNoise

def soft_update(target, source, tau):
    """ Perform soft update"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class DDPGAgent:
    """
    DDPG Agent, valid for continuous actioin space
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()
    iter = 0

    def __init__(self, func1, func2, o_dim, a_dim, h_dim,
                 initialize_weights = False, lr_actor = 1e-3, lr_critic = 1e-3,
                 batch_size = 16, gamma = 0.99, tau = 0.001, buffer_size = int(1e5),
                 seed = 1234):

        """
        func1: actor model
        func2: critic model
        o_dim/c_dim: observation space dimension/ # of channels when image as input
        a_dim: action space dimension
        """

        self.o_dim = o_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.initialize_weights = initialize_weights

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.seed = seed

        # Replay memory
        self.buffer = ReplayBuffer(buffer_size, batch_size, seed)

        # Actor Network (w/ Target Network)
        self.actor = func1(o_dim , a_dim, h_dim, initialize_weights, seed).to(self.device)
        self.target_actor = func1(o_dim , a_dim, h_dim, initialize_weights, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = lr_actor)

        # Critic Network (w/ Target Network)
        self.critic = func2(o_dim , a_dim, h_dim, initialize_weights, seed).to(self.device)
        self.target_critic = func2(o_dim , a_dim, h_dim, initialize_weights, seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = lr_critic)

        # Noise process
        self.noise = OUNoise(a_dim)

    def get_action1(self, state, eps = 0.):
        """
        action value ranges from -1 to 1
        --
        eps = 0. no exploration
            > 0. add exploration
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor)[0].detach().cpu().numpy()
        self.actor.train()

        action += self.noise.sample() * eps

        return np.clip(action, -1, 1)

    def get_action2(self, state, eps = 0.):
        """
        slimevolly gym environment
        ---
        multibinary action space (although the action space is multi-binary, float vectors are accepted)
        forward = True if action[0]>0 else False
        backward = True if action[1]>0 elseTrue False
        jump = True if action[2]>0 else True False
        --
        eps = 0. no exploration
            > 0. add exploration
        """
        if random.random() > eps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.actor.eval()
            with torch.no_grad():
                logits = self.actor(state_tensor).squeeze()
                action = torch.where(logits>0,torch.ones_like(logits),torch.zeros_like(logits))
            self.actor.train()
            return action.detach().cpu().numpy()

        else:
            action = [random.choice([0,1]) for _ in range(self.a_dim)]
            return np.asarray(action, dtype = np.float32)


    def update(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        dones = torch.FloatTensor(dones).view(-1, 1).to(self.device)
        self.iter += 1
        # ---------------------------- update critic ---------------------------- #
        next_actions = self.target_actor(next_states)
        Q_next = self.target_critic(next_states, next_actions)
        Q_targets = rewards + self.gamma * Q_next * (1. -dones)
        Q_expected = self.critic(states, actions)
        critic_loss = self.loss_fn(Q_expected, Q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_targets(self):
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        if (len(self.buffer) > self.batch_size):
            experiences = self.buffer.sample()
            self.update(experiences)
            self.update_targets()
            self.iter += 1

    def reset(self):
        self.noise.reset()
