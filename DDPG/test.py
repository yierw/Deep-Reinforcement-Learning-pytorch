import gym
gym.logger.set_level(40)
import numpy as np
from collections import namedtuple, deque
from IPython import display
import matplotlib.pyplot as plt

from ddpg_agent import *

def train_agent(n_episodes = 100, print_every = 10, tmax = 1000):
    scores = [];
    score_window = deque(maxlen = print_every)

    best_score = 0.0
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        sum_rewards = 0
        for t in range(tmax):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            sum_rewards += reward
            if done:
                break

        scores.append(sum_rewards)
        score_window.append(sum_rewards)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(score_window)), end="")

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(score_window)))
            if np.mean(score_window) >= best_score:
                print('\nmodel saved!')
                torch.save(agent.online_net.state_dict(), 'checkpoint.pth')
                best_score  = np.mean(score_window)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
################################################################################

env = gym.make('BipedalWalker-v3')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
print('State shape: ', state_size)
print('Action shape: ', action_size)
agent = Agent(state_size, action_size)

is_train = True
is_watch = True

# train agent
if is_train:
    train_agent(n_episodes = 700, print_every = 50)
# watch agent
if is_watch:
    #agent.online_net.load_state_dict(torch.load('checkpoint.pth'))
    for _ in range(3):
        state = env.reset()
        img = plt.imshow(env.render(mode='rgb_array'))
        for _ in range(1000):
            action = agent.get_action(state, add_noise = False)
            img.set_data(env.render(mode='rgb_array'))
            plt.axis('off')
            display.display(plt.gcf())
            display.clear_output(wait=True)
            state, reward, done, _ = env.step(action)
            if done:
                break

env.close()
