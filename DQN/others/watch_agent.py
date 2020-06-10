### !brew install ffmpeg
import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
# gymlogger.set_level(40)

import io
import math
import glob
import random
import numpy as np

import base64
from IPython.display import HTML
from IPython import display as ipythondisplay

from DQN_agent import *

def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env


#======================
env = wrap_env( gym.make('LunarLander-v2'))
env.seed(1234)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size, action_size)

# watch a trained agent
agent.online_net.load_state_dict(torch.load('cpu_checkpoint.pth'))

observation = env.reset()

while True:
    env.render()
    action = agent.act(observation, 0.)
    #action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break

env.close()
#show_video()
