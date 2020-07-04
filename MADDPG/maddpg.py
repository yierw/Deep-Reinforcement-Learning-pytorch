import torch
from ddpg import DDPGAgent
from buffer import ReplayBuffer

def soft_update(target, source, tau):
    """ Perform soft update"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class MADDPGAgent:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iter = 0

    def __init__(self, num_agents, x_dim, o_dim, a_dim, lr_actor = 1e-3, lr_critic = 1e-3,
                 batch_size = 16, gamma = 0.99, tau = 0.001, buffer_size = int(1e5), seed = 1234):

        self.num_agents = num_agents
        self.x_dim = x_dim
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.seed = seed

        self.buffer = ReplayBuffer(buffer_size, batch_size, seed)
        self.agents = [DDPGAgent(num_agents, id, x_dim, o_dim, a_dim, lr_actor, lr_critic, gamma, seed) \
                       for id in range(num_agents)]

    def get_actions(self, obs_full, eps = 0.):
        """get actions from all agents in the MADDPG object"""
        actions = []
        for id, agent in enumerate(self.agents):
            actions.extend(agent.get_action2(obs_full[id,:], eps) )
        return actions


    def update(self, experiences):

        obs_full, actions, rewards, next_obs_full, dones = experiences

        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        x = torch.FloatTensor(obs_full).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        next_x = torch.FloatTensor(next_obs_full).to(self.device)

        with torch.no_grad():
            next_a = [agent.target_actor(next_x[:, agent.id, :]) for agent in self.agents]
        next_a = torch.cat(next_a, dim=1)

        for agent in self.agents:
            r = rewards[:, agent.id].view(-1, 1)
            d = dones[:, agent.id].view(-1, 1)

            pred_a = [ self.agents[i].actor(x[:, i, :]) if i == agent.id \
                       else self.agents[i].actor(x[:, i, :]).detach()
                       for i in range(self.num_agents) ]
            pred_a  = torch.cat(pred_a , dim=1)

            agent.update(next_x, next_a, r, d, x, a, pred_a)

    def update_targets(self):
        """soft update targets"""
        for agent in self.agents:
            soft_update(agent.target_actor, agent.actor, self.tau)
            soft_update(agent.target_critic, agent.critic, self.tau)

    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        if (len(self.buffer) > self.batch_size):
            experiences = self.buffer.sample()
            self.update(experiences)
            self.update_targets()
            self.iter += 1

    def reset(self):
        for agent in self.agents:
            agent.noise.reset()
