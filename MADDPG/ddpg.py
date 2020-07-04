import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import Critic, Actor
from OUNoise import OUNoise

class DDPGAgent:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()

    def __init__(self, num_agents, id, x_dim, o_dim, a_dim, lr_actor, lr_critic, gamma, seed):

        self.id = id
        self.x_dim = x_dim
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        # Actor Network (w/ Target Network)
        self.actor = Actor(o_dim, a_dim, seed).to(self.device)
        self.target_actor = Actor(o_dim, a_dim, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr = lr_actor)
        # Critic Network (w/ Target Network)
        self.critic = Critic(x_dim, num_agents * a_dim, seed).to(self.device)
        self.target_critic = Critic(x_dim, num_agents * a_dim, seed).to(self.device)
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
            action = self.actor(state_tensor)[0].detach().cpu().numpy() + self.noise.sample() * eps
        self.actor.train()

        return np.clip(action, -1, 1)

    def get_action2(self, state, eps = 0.):
        """
        slimevolly gym environment
        ---
        multibinary action space (although the action space is multi-binary, float vectors are accepted)
        forward = True if action[0]>0 else False
        backward = True if action[1]>0 else False
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

    def update(self, next_x, next_a, r, d, x, a, pred_a):
        x = x.flatten(start_dim = 1)
        next_x = next_x.flatten(start_dim = 1)
        # ---------------------------- update critic ---------------------------- #
        Q_next = self.target_critic(next_x, next_a)
        Q_targets = r + self.gamma * Q_next * (1. - d)
        Q_expected = self.critic(x, a)
        critic_loss = self.loss_fn(Q_expected, Q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        actor_loss = -self.critic(x, pred_a).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
