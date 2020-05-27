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

def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env


#======================
env = wrap_env( gym.make('LunarLander-v2'))
env.seed(1234)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print('State shape: ', state_size)
print('Number of actions: ', action_size)

agent = Agent(state_size, action_size)

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
