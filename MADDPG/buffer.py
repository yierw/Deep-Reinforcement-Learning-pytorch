import random
import numpy as np
from collections import namedtuple, deque

class ReplayBuffer:
    """revised replay buffer for multi agent system"""

    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __init__(self, size = int(1e5), batch_size = 64, shared_obs = False, seed = 1234) :
        self.size = size
        self.batch_size = batch_size
        self.shared_obs = shared_obs
        self.seed = random.seed(seed)
        self.memory = deque(maxlen = size)

    def push(self, state, action, reward, next_state, done):
        """push new experience(s) to memory"""

        reward_copy, done_copy = reward, done

        if type(reward) is list:
            pass
        else:
            n = len(action)
            reward = [reward_copy] * n

        if type(done) is list:
            done = [1 if x else 0 for x in done_copy]
        else:
            n = len(action)
            done = [1 if done_copy else 0] * n

        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        rewards (array): (num_agent, batch_size)
        dones (array):  (num_agent, batch_size)
        actions_i (array): (batch_size, action_dim)
        actions (list): [..., actions_i, ...]

        shared_obs = True
        states (array): (batch_size, state_dim)

        shared_obs = False
        states_i (array): (batch_size, state_dim)
        states (list): [..., states_i, ...]
        """

        samples = random.sample(self.memory, k = self.batch_size)
        batch = self.experience(*zip(*samples))

        rewards = np.asarray(batch.reward).T
        dones = np.asarray(batch.done).T

        actions = [np.asarray(ai) for ai in zip(*batch.action)]

        if self.shared_obs:
            states = np.asarray(batch.state)
            next_states = np.asarray(batch.next_state)
        else:
            states = [np.asarray(si) for si in zip(*batch.state)]
            next_states = [np.asarray(si) for si in zip(*batch.next_state)]

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
