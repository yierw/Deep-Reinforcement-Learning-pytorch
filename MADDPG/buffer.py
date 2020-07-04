import random
import numpy as np
from collections import namedtuple, deque

class ReplayBuffer:

    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __init__(self, size = int(1e5), batch_size = 64, seed = 1234) :
        self.size = size
        self.batch_size = batch_size
        self.memory = deque(maxlen = size)

        random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        """push new experience(s) to memory"""
        done_copy = done

        if isinstance(done, (list, np.ndarray)):
            done = [1 if x else 0 for x in done_copy]
        else:
            done = 1 if done_copy else 0

        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        """
        Randomly sample a batch of experiences from the memory

        states (array): (batch_size, num_agents, state_dim) if num_agents > 1
                        (batch_size, state_dim)             if num_agents = 1

        actions (array): (batch_size, action_dim * num_agents)

        rewards (array): (batch_size, num_agents)     if num_agents > 1
                         (batch_size, )               if num_agents = 1

        dones (array):  (batch_size, num_agents)      if num_agents > 1
                        (batch_size, )                if num_agents = 1


        """

        samples = random.sample(self.memory, k = self.batch_size)
        batch = self.experience(*zip(*samples))

        states = np.asarray(batch.state)
        actions = np.asarray(batch.action)
        rewards = np.asarray(batch.reward)
        next_states = np.asarray(batch.next_state)
        dones = np.asarray(batch.done)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
