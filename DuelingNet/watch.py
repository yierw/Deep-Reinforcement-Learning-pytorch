### !brew install ffmpeg
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

def preprocess_frames(frame_list):
    x = np.asarray(frame_list) # stack frames
    x = x[:,34:-16:2,::2,:] # crop and downsampling
    x = np.mean(x, axis = -1)/255.  # convert to grey scale
    return x

def act(env, action):
    frame_list = deque(maxlen = 2)
    reward = 0
    for a in decoded_dict[action]:
        frame, r, done, _ = env.step(a)
        frame_list.append(frame)
        reward += r
    next_state = preprocess_frames(frame_list)
    return next_state, reward, done

env = wrap_env( gym.make('EnduroDeterministic-v4'))
env.seed(1234)

decoded_dict = {c:[c] + [0]*3 for c in range(9)}
decoded_dict.update({7 + c:[c]*2 + [0]*2 for c in range(2,9)})
decoded_dict.update({14 + c:[c]*3 + [0] for c in range(2,9)})
decoded_dict.update({21 + c:[c]*4 for c in range(2,9)})
a_dim = len(decoded_dict)
c_dim = 2
agent = DQNAgent(DuelingNet, c_dim, a_dim, lr = 1e-3, batch_size = 32)

agent.online_net.load_state_dict(torch.load('model_weights/Enduro.pth'))

observation = env.reset()

score = 0
env.reset()
state, reward, done = act(env, 0)
score += reward
while True:
    action = agent.get_action(state, 0)
    next_state, reward, done = act(env, action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
            break

env.close()
