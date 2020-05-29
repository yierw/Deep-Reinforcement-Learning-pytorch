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

from dueling_agent import *

#from DQN_agent import *

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
env = wrap_env( gym.make('Enduro-v0'))
env.seed(1234)

action_dim = env.action_space.n
agent = Agent(action_dim)

# watch a trained agent
agent.online_net.load_state_dict(torch.load('cpu_checkpoint.pth'))

k = 4
sum_rewards = 0
# get initial state
frame_list = deque(maxlen = k)
frame_list.append(env.reset())
for _ in range(k-1):
    action = env.action_space.sample()
    frame, reward, done, _ = env.step(action)
    sum_rewards += reward
    frame_list.append(frame)
state = preprocess_frames(frame_list)

while True:
    action = agent.act(state)
    frame, reward, done, _ = env.step(action)
    sum_rewards += reward
    print('\rsum of rewards {}'.format(sum_rewards), end="")
    frame_list.append(frame)
    next_state = preprocess_frames(frame_list)
    #agent.step(state, action, reward, next_state, done)
    state = next_state
    if done:
        break
print('\rsum of rewards {}'.format(sum_rewards))
env.close()
#show_video()
