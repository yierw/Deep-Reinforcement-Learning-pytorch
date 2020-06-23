import random
import numpy as np
from collections import namedtuple, deque

from sumtree import SumTree

class PrioritizedBuffer:

    experience = namedtuple("Experience", field_names=["index","IS_weight","state", "action", "reward", "next_state", "done"])
    alpha = 0.6 # mixing pure greedy prioritization and uniform random sampling
    beta = 0.4 # compensate for the non-uniform probabilities
    beta_increment_per_sampling = 0.001
    epsilon = 0.01 # small amount to avoid zero priority
    current_length = 0

    def __init__(self, size = int(1e5), batch_size = 64, seed = 1234) :
        self.size = size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = SumTree(capacity = self.size)

    def push(self, state, action, reward, next_state, done):
        """push new experience(s) to memory"""
        max_p = self.memory.tree[-self.memory.capacity:].max()
        priority = 1.0 if self.current_length == 0 else max_p
        data = (state, action, reward, next_state, done)
        self.memory.add(priority, data)
        self.current_length = self.current_length + 1

    def sample(self):
        sum_priority = self.memory.total()
        segment = sum_priority/self.batch_size
        samples = []
        for i in range(self.batch_size):
            a, b = segment * i, segment *(i+1)
            s = random.uniform(a,b)
            (idx, priority, data) = self.memory.get(s)
            p = priority/sum_priority
            IS_weight = (self.batch_size * p)**(-self.beta)
            samples.append(self.experience(idx, IS_weight, data[0], data[1], data[2], data[3], data[4]))

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        batch = self.experience(*zip(*samples))

        index = np.asarray(batch.index)

        IS_weight = np.asarray(batch.IS_weight)
        max_weight = IS_weight.max()
        IS_weight = IS_weight/max_weight

        states = np.asarray(batch.state)
        actions = np.asarray(batch.action)
        rewards = np.asarray(batch.reward)
        next_states = np.asarray(batch.next_state)
        dones = np.asarray(batch.done).astype(np.uint8)

        return (index, IS_weight, states, actions, rewards, next_states, dones)

    def update_priority(self, idxs, td_errors):
        """update priority for the replayed transitions"""
        for idx, td_error in zip(idxs, td_errors):
            priority = (td_error + self.epsilon)**self.alpha
            self.memory.update(idx, priority)

    def __len__(self):
        return self.current_length
