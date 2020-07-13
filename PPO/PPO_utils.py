from parallelEnv import parallelEnv

import numpy as np

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RIGHT=4
LEFT=5

def get_action(policy, state):
    """
    input:
        policy
        state (array)
    output:
        action (array)
        prob (array): prob of selecting action
    """
    state = torch.from_numpy(state).float().to(device)
    out = policy(state).squeeze().cpu().detach().numpy()
    action = np.where(np.random.rand(state.size(0)) < out, RIGHT, LEFT)
    prob = np.where(action == RIGHT, out, 1.0-out)
    return action, prob

def get_prob(policy, state, action):
    """
    input:
        policy
        state (array)
        action (array)
    output:
        prob (tensor): prob of selecting action
    """
    action = torch.from_numpy(action).long().to(device).view(-1,1)
    state = torch.from_numpy(state).float().to(device)
    out = policy(state)
    prob = torch.where(action == RIGHT, out, 1.0-out)
    return prob


def preprocess_frames(frame_list, bkg_color = np.array([144, 72, 17])):
    x = np.asarray(frame_list)[:,34:-16:2,::2,:]-bkg_color
    x = np.mean(x, axis=-1)/255.
    return x[np.newaxis,:,:,:]


def preprocess_frames_batch(frame_list, bkg_color = np.array([144, 72, 17])):
    x = np.asarray(frame_list)[:,:,34:-16:2,::2,:]-bkg_color
    x = np.mean(x, axis=-1)/255.
    return np.swapaxes(x,0,1)


def collect_trajectories(envs, policy, tmax):
    """ collect trajectories for a parallelized parallelEnv object """
    n=len(envs.ps)
    policy.eval()

    state_list, reward_list, prob_list, action_list = [], [], [], []

    # initialize
    fr1 = envs.reset()
    fr2, _, _, _ = envs.step([1]*n)
    for t in range(tmax):
        state = preprocess_frames_batch([fr1,fr2])
        action, prob = get_action(policy, state)
        fr1, re1, done, _ = envs.step(action)
        if done.any():
            break
        fr2, re2, done, _ = envs.step([0]*n)
        if done.any():
            break

        state_list.append(state)      # array
        action_list.append(action)    # array
        prob_list.append(prob)        # array
        reward_list.append(re1 + re2) # array

    return state_list, action_list, prob_list, reward_list


def r2g(reward_list, gamma = 0.99):
    """compute normalized sum of discounted rewards to go """
    reward_array = np.asarray(reward_list)
    T = len(reward_list)
    R = []
    for t in range(T):
        # sum of discounted rewards to go
        discounts = gamma**np.arange(T-t)
        reward_to_go = reward_array[t:,:]*discounts[:,np.newaxis]
        sum_of_rewards = reward_to_go.sum(axis = 0) # sum of steps
        # normalization
        mean, std= np.mean(sum_of_rewards), np.std(sum_of_rewards) + 1.0e-10
        normalized = (sum_of_rewards - mean)/std
        R.append(torch.from_numpy(normalized).float().view(1,-1))
    return torch.cat(R)
