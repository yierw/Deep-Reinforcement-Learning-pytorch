import os
import imageio

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv

from maddpg import MADDPGAgent

def make_env(scenario_name,benchmark=False):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

env = make_env(scenario_name="simple_adversary")
print('{} agents in thie environment'.format(env.n))
print(env.observation_space)
print(env.action_space)

folder = os.getcwd()+"/video"
os.makedirs(folder, exist_ok = True)

maddpg = MADDPGAgent()

frames = []
obs_full = env.reset()
for _ in range(300):
    actions = maddpg.get_actions(obs_full)
    next_obs_full, rewards, dones, info = env.step(actions)
    frames.append(env.render('rgb_array')[0])

obs_full = next_obs_full

imageio.mimsave(os.path.join(folder, 'simple_adversary.gif'),frames, duration=.04)
