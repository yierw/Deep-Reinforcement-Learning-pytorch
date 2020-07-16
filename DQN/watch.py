import numpy as np
from collections import deque

import gym
from gym.wrappers import Monitor

import base64
from IPython.display import HTML
from IPython import display as ipythondisplay

from agent import *
from model import *

def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env

env = wrap_env( gym.make('LunarLander-v2'))

o_dim = env.observation_space.shape[0]
a_dim = env.action_space.n
print('State shape: ', o_dim)
print('Number of actions: ', a_dim)
agent = DQNAgent(QNetwork, o_dim, a_dim)
agent.online_net.load_state_dict(torch.load('ddqn_LunarLander.pth'))

state = env.reset()
score = 0
while True:
    action = agent.get_action(state)
    state, reward, done, _ = env.step(action)
    score += reward
    if done:
        break
print("score:",score)

env.close()
