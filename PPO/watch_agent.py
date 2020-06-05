### !brew install ffmpeg
from model import Policy
from PPO_utils import *

import gym
from gym.wrappers import Monitor
from gym import logger as gymlogger
gymlogger.set_level(40)

def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env

env = wrap_env( gym.make('PongDeterministic-v4'))
env.seed(1234)

#RIGHT, LEFT = 4, 5
#observation = env.reset()
#for t in range(1000):
#    env.render()
#    observation, reward, done, info = env.step(np.random.choice([RIGHT,LEFT]))
#    if done:
#        break
#    observation, reward, done, info = env.step(0)
#    if done:
#        break
#print(t)

# load trained policy
policy = Policy()
policy.load_state_dict(torch.load('model_weights/pong_PPO.pth'))
policy.eval()

fr1 = env.reset()
fr2, _, _, _ =env.step(1)
for t in range(1000):
    env.render()
    state = preprocess_frames([fr1 , fr2])
    action, _ = get_action(policy, state)
    fr1, re1, done, _ = env.step(action)
    if done:
        break
    fr2, re2, done, _ = env.step(0)
    if done:
        break
print(t)
env.close()
