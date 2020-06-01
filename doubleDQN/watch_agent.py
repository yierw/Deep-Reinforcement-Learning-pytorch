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

from DDQN_agent import *

def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env

###
env = wrap_env( gym.make('Enduro-v0'))
env.seed(1234)

action_dim = env.action_space.n
NUM_FRAMES = 4

agent = Agent(action_dim, NUM_FRAMES)
agent.online_net.load_state_dict(torch.load('trained_weights/cpu_checkpoint.pth'))
agent.online_net.eval()

# initialize
env.reset()
sum_rewards = 0
frame_list = deque(maxlen = NUM_FRAMES)

for _ in range(NUM_FRAMES):
    action = env.action_space.sample()
    frame, reward, done, _ = env.step(action)
    sum_rewards += reward
    frame_list.append(frame)

state = preprocess_frames(frame_list)

while True:
    action = agent.act(state)
    reward, next_state, done = collect_tuple(env, action, k = NUM_FRAMES)
    state = next_state
    sum_rewards += reward
    print('\rsum of rewards {}'.format(sum_rewards), end="")
    if done:
        break
print('\rsum of rewards {}'.format(sum_rewards))

env.close()
