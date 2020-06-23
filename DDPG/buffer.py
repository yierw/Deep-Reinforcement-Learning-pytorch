import random
import numpy as np
from collections import namedtuple, deque

class ReplayBuffer:

    experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __init__(self, size = int(1e5), batch_size = 64, seed = 1234) :
        self.size = size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen = size)

    def push(self, state, action, reward, next_state, done):
        """push new experience(s) to memory"""
        done_copy = done
        done = 1 if done_copy else 0
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        rewards (array): (batch_size,)
        dones (array):  (batch_size,)
        actions (array): (batch_size, action_dim)
        states (array): (batch_size, state_dim)
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
