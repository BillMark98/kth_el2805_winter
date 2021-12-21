# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518
#  
import os
from numpy.random.mtrand import random
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import trange

from DDPG_agent import DDPG_agent, RandomAgent


import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))



# # load the parameters

# # Load model
# try:
#     model = torch.load('neural-network-1.pth')
#     print('Network model: {}'.format(model))
# except:
#     print('File neural-network-1.pth not found!')
#     exit(-1)


# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

y_Nums = 100
omega_Nums = 100

# color map scheme
value_func_cmap = cm.coolwarm
# action_cmap = cm.coolwarm
action_cmap = "binary"



def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# # Load model
# try:
#     model = torch.load('neural-network-1.pth')
#     print('Network model: {}'.format(model))
# except:
#     print('File neural-network-1.pth not found!')
#     exit(-1)

# Parameters
N_EPISODES = 50            # Number of episodes to run for trainings
CONFIDENCE_PASS = 50


def simulate(env, agent, episodes, agentName = "DDPG"):
    ''' simulate '''

    # Reward
    episode_reward_list = []  # Used to store episodes reward

    # Simulate episodes
    print('Using ' + agentName)
    EPISODES = trange(episodes, desc='Episode: ', leave=True)
    for i in EPISODES:
        EPISODES.set_description("Episode {}".format(i))
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        while not done:
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            action = agent.forward(state)
            # env.render()
            next_state, reward, done, _ = env.step(action)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()

    avg_reward = np.mean(episode_reward_list)
    confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)


    print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
                    avg_reward,
                    confidence))
    return episode_reward_list

# create two agents
ddpg_agent = DDPG_agent(n_actions=2,loadPrev=True)
random_agent = RandomAgent(n_actions=2)
ddpg_episode_reward_list = simulate(env, ddpg_agent, N_EPISODES, "DDPG")
random_episode_reward_list = simulate(env, random_agent, N_EPISODES, "Random")

# plot the rewards
# save Name
saveName = "Random_DDPG_rewards.png"
plt.plot(list(range(1,len(ddpg_episode_reward_list) + 1)), ddpg_episode_reward_list, label = 'ddpg')
plt.plot(list(range(1,len(random_episode_reward_list) + 1)), random_episode_reward_list, label = 'random')
plt.title("Random vs DDPG")
plt.ylabel("reward")
plt.xlabel("episode")
plt.legend()
print("plot saved as " + saveName)
plt.savefig(saveName)
plt.show()
