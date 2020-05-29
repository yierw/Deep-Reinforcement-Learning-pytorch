import gym
import numpy as np
from collections import namedtuple, deque
from IPython import display
import matplotlib.pyplot as plt

from dueling_agent import *

def train_agent(n_episodes = 100, print_every = 10, tmax = 1000, eps_start = 1.0, eps_end = 0.01, eps_decay = 0.995):
    scores = [];
    score_window = deque(maxlen = print_every)
    eps = eps_start
    best_score = 0.0
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        sum_rewards = 0
        for t in range(tmax):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            sum_rewards += reward
            if done:
                break
        # decrease epsilon every eposide
        eps = max(eps_end, eps_decay * eps)
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

env = gym.make('Enduro-v0')
action_dim = env.action_space.n

agent = Agent(action_dim)

is_train = False
is_watch = True

train_agent(n_episodes = 10, print_every = 1)
