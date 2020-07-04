import random
import numpy as np
from collections import deque

import torch
import torch.optim as optim
from torch.distributions import Categorical

from model import TwoHeadNetwork


def soft_update(target, source, tau):
    """ Perform soft update"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Agent:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.SmoothL1Loss()
    iter = 0
    def __init__(self, o_dim, a_dim, lr = 1e-3, tmax = 4, gamma = 0.99, \
                 tau = 0.001, update_every = 1, seed = 1234):

        """
        func1: actor model
        func2: critic model
        o_dim/c_dim: observation space dimension/ # of channels when image as input
        a_dim: action space dimension
        """

        self.o_dim = o_dim
        self.a_dim = a_dim
        self.lr = lr
        self.tmax = tmax
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every

        self.state_list = deque(maxlen = tmax)
        self.action_list = deque(maxlen = tmax)
        self.reward_list = deque(maxlen = tmax)

        # Shared Actor-Critic Network (w/ Target Network)
        self.online_net = TwoHeadNetwork(o_dim , a_dim, seed).to(self.device)
        self.target_net = TwoHeadNetwork(o_dim , a_dim, seed).to(self.device)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr = self.lr)
        soft_update(self.target_net, self.online_net, 1.)

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            prob, _ = self.online_net(state_tensor)
            m = Categorical(prob)
            action = m.sample()
        self.online_net.train()
        return action.detach().cpu().tolist()


    def update(self, states, actions, reward_list, next_states, dones):

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        actions = torch.LongTensor(actions).view(-1, 1).to(self.device)
        dones = torch.FloatTensor(dones).view(-1, 1).to(self.device)

        discounts = self.gamma**np.arange(len(reward_list))
        rewards = np.asarray(reward_list)*discounts[:,np.newaxis]
        rewards = torch.FloatTensor(rewards.sum(axis = 0))
        # ---------------------------- critic loss ---------------------------- #
        _, value_next = self.target_net(next_states)
        V_targets = rewards.unsqueeze(1) + value_next
        prob, V_expected = self.online_net(states)
        critic_loss = self.loss_fn(V_expected, V_targets.detach())
        # ---------------------------- actor loss ---------------------------- #
        logp = torch.log(prob.gather(1, actions))
        actor_loss = -logp * (V_targets.detach() - V_expected.detach())
        actor_loss = actor_loss.mean()
        # ---------------------------- entropy regularization ---------------------------- #
        #entropy = [-torch.sum(p * torch.log(p)) for p in prob]
        #entropy_term = torch.stack(entropy).mean()

        # ---------------------------- update parameters ---------------------------- #
        loss = actor_loss + critic_loss #+ 0.001 * entropy_term

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.)
        self.optimizer.step()


    def step(self, state, action, reward, next_state, done):

        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)

        done_copy = done
        if isinstance(done, (list, np.ndarray)):
            done = [1 if x else 0 for x in done_copy]
        else:
            done = 1 if done_copy else 0

        if len(self.reward_list) >= self.tmax:
            self.update(self.state_list[0], self.action_list[0], self.reward_list, next_state, done)
            self.iter += 1

        if self.iter % self.update_every == 0:
            soft_update(self.target_net, self.online_net, self.tau)
