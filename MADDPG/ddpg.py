import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import CentralizedCritic, Actor
from OUNoise import OUNoise



class DDPGAgent:

    def __init__(self, x_dim, all_a_dim, obs_dim, action_dim, lr_actor, lr_critic, device):
        super(DDPGAgent, self).__init__()
        self.x_dim = x_dim
        self.all_a_dim = all_a_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device
        self.loss_fn = nn.MSELoss()
        # Actor Network (w/ Target Network)
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.target_actor = Actor(obs_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = lr_actor )
        # Critic Network (w/ Target Network)
        self.critic = CentralizedCritic(x_dim, all_a_dim).to(device)
        self.target_critic = CentralizedCritic(x_dim, all_a_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = lr_critic)
        # Noise process
        self.noise = OUNoise(action_dim, scale = 1.0)

    def get_action(self, obs, noise = 0.0):
        obs = torch.from_numpy(obs).float().view(1,-1).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs)[0].detach().cpu().numpy() + noise * self.noise.sample()

        self.actor.train()

        return np.clip(action, -1, 1)

    def update(self, x, a, next_x, next_a, pred_a, done, r, gamma):
        # ---------------------------- update critic ---------------------------- #
        Q_next = self.target_critic(next_x, next_a)
        Q_targets = r + gamma * Q_next * (1 - done)
        Q_expected = self.critic(x, a)
        critic_loss = self.loss_fn(Q_expected, Q_targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        actor_loss = -self.critic(x, pred_a).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def target_update(self, tau):
        soft_update(self.target_critic, self.critic, tau)
        soft_update(self.target_actor, self.actor, tau)

    def reset(self):
        self.noise.reset()


def soft_update(target, source, tau):
    """ Perform soft update"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
