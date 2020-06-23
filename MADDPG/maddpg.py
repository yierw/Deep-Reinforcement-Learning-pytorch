import torch
from buffer import ReplayBuffer
from ddpg import DDPGAgent


class MADDPGAgent:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iter = 0
    t_step = 0

    def __init__(self, obs_dim_list, action_dim_list, shared_obs = False,
                 lr_actor = 0.01, lr_critic = 0.01, batch_size = 64, gamma = 0.95, tau = 0.1, buffer_size = int(1e6), update_every = 10):
        super(MADDPGAgent, self).__init__()
        """
        shared_obs = True: all agents see the same observation; x_dim = obs_dim
        shared_obs = False: agents see differenct observation; x_dim = sum(obs_dim_list)
        """
        self.shared_obs = shared_obs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        if shared_obs:
            self.x_dim = obs_dim_list[0]
        else:
            self.x_dim = sum(obs_dim_list)
        self.all_a_dim = sum(action_dim_list)

        self.agents = [DDPGAgent(self.x_dim, self.all_a_dim, o_dim, a_dim, lr_actor, lr_critic, self.device) for o_dim, a_dim in zip(obs_dim_list, action_dim_list) ]

        self.num_agents = len(self.agents)

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.buffer = ReplayBuffer(buffer_size, self.batch_size, self.shared_obs)
        self.update_every = update_every

    def get_actions(self, obs_full, noise = 0.0):
        """get actions from all agents in the MADDPG object"""
        if self.shared_obs:
            return [agent.get_action(obs_full, noise) for agent in self.agents]
        else:
            return [agent.get_action(obs, noise) for agent, obs in zip(self.agents, obs_full)]

    def update(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        if self.shared_obs:

            x = torch.from_numpy(states).float()
            a = torch.cat([torch.from_numpy(a).float() for a in actions], dim = 1)
            next_x = torch.from_numpy(next_states).float()
            next_a = torch.cat([agent.target_actor(next_x) for agent in self.agents], dim = 1)
            for i in range(self.num_agents):
                pred_a = torch.cat([agent.actor(x) for agent in self.agents], dim = 1)
                done = torch.from_numpy(dones[i]).float().view(-1, 1)
                r = torch.from_numpy(rewards[i]).float().view(-1, 1)

                self.agents[i].update(x, a, next_x, next_a, pred_a, done, r, self.gamma)
                self.agents[i].target_update(self.tau)

        else:

            x_list = [torch.from_numpy(obs).float() for obs in states]
            x = torch.cat(x_list, dim = 1)

            a = torch.cat([torch.from_numpy(a).float() for a in actions], dim = 1)

            next_x_list = [torch.from_numpy(next_obs).float() for next_obs in next_states]
            next_x = torch.cat(next_x_list, dim = 1)

            next_a = [agent.target_actor(next_obs) for agent, next_obs in zip(self.agents, next_x_list)]
            next_a = torch.cat(next_a, dim = 1)

            for i in range(self.num_agents):
                pred_a = torch.cat([agent.actor(obs) for agent, obs in zip(self.agents, x_list)], dim = 1)
                done = torch.from_numpy(dones[i]).float().view(-1, 1)
                r = torch.from_numpy(rewards[i]).float().view(-1, 1)

                self.agents[i].update(x, a, next_x, next_a, pred_a, done, r, self.gamma)
                self.agents[i].target_update(self.tau)


    def step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        if (len(self.buffer) <= self.batch_size):
            pass
        else:
            if self.t_step % self.update_every == 0:
                experiences = self.buffer.sample()
                self.update(experiences)
                self.iter += 1

            self.t_step += 1

    def reset(self):
        for agent in self.agents:
            agent.noise.reset()
