# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518
#  
# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
from numpy.lib.function_base import gradient
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
from DQN_agent import DQN_NeuronNetwork, DQN_agent
from ReplayBuffer import ExperienceReplayBuffer, Experience
import torch.optim as optim
import random

import os

def change2FileDir():
    """ change working directory to the current file directory"""
    # change to the file position
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

change2FileDir()

# copy module
import copy
# from lab.lab2.problem1.DQN_agent import DQN_NeuronNetwork

# from lab.lab2.problem1.ReplayBuffer import Experience

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

# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 1000                             # Number of episodes
discount_factor = 0.99                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
agent = RandomAgent(n_actions)

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

eps_max = 0.99
eps_min = 0.05
Z = np.floor(0.9 * N_episodes)



# method to decay epsilon
eps_decay_method = 2

# def epsilonDecay(method = eps_decay_method):
#     ''' the function to update the epsilon, function of episode '''
#     def eps_decay(episode):
#         if (method == 1):
#             episode = np.max([eps_min, eps_max - ((eps_max - eps_min) * (episode - 1) / (Z - 1))])
#         else:
#             episode = np.max([eps_min, eps_max * np.float_power(eps_min / eps_max, (episode - 1) / (Z - 1))])
#         return episode

#     return eps_decay



# n1 = np.int32(np.sqrt((n_actions + 2) * N) + 2 * np.sqrt(N / (n_actions + 2)))
# n2 = np.int32(n_actions * np.sqrt(N & (n_actions + 2)))

# ****************************************
# ****************************************
# !! set parameters here!!


# discount factor
gamma = 1
# buffer_size
L = 10000
# mini-batch size
N = 60
# target update frequency
C = L // N
# learning rate
alpha = 0.0005

n1 = 80
n2 = 80
# neuron network design
neuron_layers = 2
neuron_num_per_layer = [n1, n2]

# to clip gradient
gradient_clip = True
# optimizer clipping of gradient between 0.5 - 2
clipping_value = 1.

init_fill_fraction = 0.2
CER = True
dueling = True
adaptiveC = False
preExit_episodeMin = 300
# prematurely stop if average reward above a threshold
premature_stop = False
# load previous learned model, instead of learning from scratch
loadPrev = False
modelFileName = 'neural-network-1.pth'

# ****************************************
# ****************************************

# suffix = "a_5e4_nn_{0}_{1}_".format(n1,n2) + "cer_1" + "duel_1_newDes_preExit_adapC_"
# suffix = "a_5e4_nn_{0}_{1}_".format(n1,n2) + "cer_1" + "duel_1_newDes_preExit_"
suffix = "a_5e4_nn_{0}_{1}_".format(n1,n2) + "cer_1" + "duel_1_newDes_tillEnd_loadPrev_"
suffix += "fillFract_{:.0e}".format(init_fill_fraction)
suffix += "loadPrev_{0}".format(loadPrev)
suffix += "gamma_{0:.2e}".format(gamma)




# def DQN_learn(env, eps_decay_Func = epsilonDecay(eps_decay_method), 
#     discount_factor = gamma, buffer_size = L,  train_batch_size = N,
#     episodes = EPISODES, target_freq_update = C, learning_rate = alpha, 
#     neuron_layers = neuron_layers, neuronNums = neuron_num_per_layer, clipping_value = clipping_value,
#     exp_cer = True, dueling = dueling):
def DQN_learn(env, eps_decay_method = eps_decay_method, 
    discount_factor = gamma, buffer_size = L,  train_batch_size = N,
    init_fill_fraction = init_fill_fraction,
    episodes = EPISODES, target_freq_update = C, learning_rate = alpha, 
    neuron_layers = neuron_layers, neuronNums = neuron_num_per_layer, clipping_value = clipping_value,
    exp_cer = True, dueling = dueling,premature_stop = premature_stop, threshold = 190, optimal_len = 50, adaptiveC=adaptiveC,
    loadPrev = loadPrev, modelFileName = modelFileName):

    ''' DQN learning algorithms
    
    ------
    Parameters
    -----

    eps_decay_Func : the function to decay epsilon

    buffer_size : the size of the experience buffer
    
    train_batch_size:  the size to sample from the buffer to train 

    episodes: trange 

    target_freq_update: the frequency that the target param is updated

    learning_rate: learning rate for the update

    clipping_value: the clipping value used for the optimizer to clip gradient

    exp_cer : boolean, to use cer for replay buffer or not

    premature_stop : stop if performs good for some time
    
    threshold : default 120
    
    optimal_len = 50        

    adaptiveC : make the C step adaptive

    loadPrev : initialize the model from previously trained, default False

    modelFileName : the network model file name
    
    -----
    return
    ------

    dictionary: 
    {
        "episode_reward_list": []
        "episode_number_of_steps" : []
    }
    '''

    # new design

    agent = DQN_agent(n_actions, eps_decay_method=eps_decay_method, discount_factor = discount_factor, buffer_size = buffer_size,
    cer = exp_cer, train_batch_size = train_batch_size, dueling = dueling, 
    init_fill_fraction = init_fill_fraction,
    episodes = N_episodes, 
    target_freq_update = target_freq_update, learning_rate = learning_rate, 
    n_inputs = 8, layers = neuron_layers, neuronNums= neuronNums, 
    gradient_clip= gradient_clip, gradient_clip_max=clipping_value, adaptiveC=adaptiveC,
    loadPrev = loadPrev, modelFileName = modelFileName)

    # initialize experience buffer
    agent.init_ExperienceBuffer(env, updateLen = 50)
    # episode reward and total number of steps lists
    episode_reward_list = []
    episode_number_of_steps = []
    result_dict = {}
    begin_optimal = False
    for episode in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        # reset agent
        epsilon = agent.reset(episode)
        total_episode_reward = 0.
        t = 0
        # epsilon = eps_decay_Func(episode)
        # moves in current episode
        move_count = 0
        while not done:
            move_count += 1
            action = agent.forward(state)
            # # create state tensor
            # state_tensor = torch.tensor(state, requires_grad=False, dtype=torch.float32)
            # # take action
            # action = agent.takeAction(epsilon, state_tensor)
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            # env.render()
            next_state, reward, done, _ = env.step(action)
            # env.render()
            agent.appendExperience(state, action, reward, next_state, done)
            # # experience created
            # exp = Experience(state, action, reward, next_state, done)
            # buffer.append(exp)
            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t+= 1

            # train the agent
            agent.backward()
            # # mini batch train NN
            # if (len(buffer) >= train_batch_size):
            #     # sample
            #     states, actions, rewards, next_states, dones = buffer.sample_batch(n = train_batch_size)

            #     # training process, set grad to 0
            #     optimizer.zero_grad()

            #     # get value
            #     Q1 = agent(torch.tensor(states, requires_grad=True,
            #                             dtype=torch.float32))
                
            #     # do not need grad
            #     with torch.no_grad():
            #         Q2 = agent_target(torch.tensor(next_states, requires_grad=False,dtype=torch.float32))
                
            #     rewards = torch.tensor(rewards, requires_grad=False,dtype=torch.float32)
            #     dones = torch.tensor(dones, requires_grad=False,dtype=torch.float32)

            #     newRewards = rewards + discount_factor * (1 - dones) * torch.max(Q2,dim=1)[0]
            #     actions = torch.tensor(actions, requires_grad=False, dtype=torch.int64)
            #     # calculated using updated model
            #     oldRewards = Q1.gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
            #     loss = loss_fn(oldRewards, newRewards.detach())
            #     loss.backward()
            #     # clip gradient
            #     nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.)
            #     optimizer.step()
            #     # update the agent target
            #     if move_count % target_freq_update == 0:
            #         agent_target.load_state_dict(agent.state_dict())


        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

        running_average_reward = running_average(episode_reward_list, n_ep_running_average)[-1]

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        # set terminal size smaller to fully print result
        EPISODES.set_description(
            "Epsilon {} ,Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            epsilon, episode, total_episode_reward, t,
            running_average_reward,
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))
        # first consider pre exit after some episodes
        if (episode > preExit_episodeMin and (premature_stop)):
            if ( (not begin_optimal) and (running_average_reward > threshold)):
                begin_optimal = True
                optimal_count_len = 0
            elif (begin_optimal and (running_average_reward > threshold)):
                optimal_count_len += 1
                if (optimal_count_len > optimal_len):
                    print("optimal len hit, exit prematurely")
                    break            
    
    # save agent
    network_name = "network_1" + suffix + ".pt"
    print("network saved as " + network_name)
    agent.save_main_nn(network_name)
    # torch.save(agent_target, network_name)

    result_dict["episode_reward_list"] = episode_reward_list
    result_dict["episode_number_of_steps"] = episode_number_of_steps
    return result_dict



    # old version, explicitly create neuron network
    # # create agent

    # # agent = RandomAgent(n_actions=n_actions, n_outputs = 8, layers = neuron_layers, neuronNums = neuronNums)
    # agent = DQN_NeuronNetwork(n_actions=n_actions, n_inputs = 8, layers = neuron_layers, neuronNums = neuronNums, dueling=dueling)
    # # copy from agent
    # agent_target = copy.deepcopy(agent)
    # agent_target.load_state_dict(agent.state_dict())
    # # create experience buffer
    # buffer = ExperienceReplayBuffer(maximum_length=buffer_size, CER = exp_cer)
    # # create optimizer
    # optimizer = optim.Adam(agent.parameters(), lr = learning_rate)
    # # loss function
    # loss_fn = torch.nn.MSELoss()

    # # episode reward and total number of steps lists
    # episode_reward_list = []
    # episode_number_of_steps = []
    # result_dict = {}

    # # initialize buffers
    # state = env.reset()

    # for i in range(buffer_size):
    #     action = np.random.randint(0,n_actions)
    #     next_state, reward, done, _ = env.step(action)
    #     # experience created
    #     exp = Experience(state, action, reward, next_state, done)
    #     buffer.append(exp)
    #     state = next_state
    #     if i % 50 == 0 :
    #         state = env.reset()
        
    # for episode in EPISODES:
    #     # Reset enviroment data and initialize variables
    #     done = False
    #     state = env.reset()
    #     total_episode_reward = 0.
    #     t = 0
    #     epsilon = eps_decay_Func(episode)
    #     # moves in current episode
    #     move_count = 0
    #     while not done:
    #         move_count += 1
    #         # create state tensor
    #         state_tensor = torch.tensor(state, requires_grad=False, dtype=torch.float32)
    #         # calculate the q value
    #         qval = agent(state_tensor)
    #         # Take a random action, epsilon greedy
    #         if (random.random() < epsilon):
    #             action = np.random.randint(0, n_actions)
    #         else :
    #             action = np.argmax(qval.data.numpy())

    #         # Get next state and reward.  The done variable
    #         # will be True if you reached the goal position,
    #         # False otherwise
    #         next_state, reward, done, _ = env.step(action)

    #         # experience created
    #         exp = Experience(state, action, reward, next_state, done)
    #         buffer.append(exp)
    #         # Update episode reward
    #         total_episode_reward += reward

    #         # Update state for next iteration
    #         state = next_state
    #         t+= 1

    #         # mini batch train NN
    #         if (len(buffer) >= train_batch_size):
    #             # sample
    #             states, actions, rewards, next_states, dones = buffer.sample_batch(n = train_batch_size)

    #             # training process, set grad to 0
    #             optimizer.zero_grad()

    #             # get value
    #             Q1 = agent(torch.tensor(states, requires_grad=True,
    #                                     dtype=torch.float32))
                
    #             # do not need grad
    #             with torch.no_grad():
    #                 Q2 = agent_target(torch.tensor(next_states, requires_grad=False,dtype=torch.float32))
                
    #             rewards = torch.tensor(rewards, requires_grad=False,dtype=torch.float32)
    #             dones = torch.tensor(dones, requires_grad=False,dtype=torch.float32)

    #             newRewards = rewards + discount_factor * (1 - dones) * torch.max(Q2,dim=1)[0]
    #             actions = torch.tensor(actions, requires_grad=False, dtype=torch.int64)
    #             # calculated using updated model
    #             oldRewards = Q1.gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
    #             loss = loss_fn(oldRewards, newRewards.detach())
    #             loss.backward()
    #             # clip gradient
    #             nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.)
    #             optimizer.step()
    #             # update the agent target
    #             if move_count % target_freq_update == 0:
    #                 agent_target.load_state_dict(agent.state_dict())


    #     # Append episode reward and total number of steps
    #     episode_reward_list.append(total_episode_reward)
    #     episode_number_of_steps.append(t)

    #     # Close environment
    #     env.close()

    #     # Updates the tqdm update bar with fresh information
    #     # (episode number, total reward of the last episode, total number of Steps
    #     # of the last episode, average reward, average number of steps)
    #     # set terminal size smaller to fully print result
    #     EPISODES.set_description(
    #         "Epsilon {} ,Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
    #         epsilon, episode, total_episode_reward, t,
    #         running_average(episode_reward_list, n_ep_running_average)[-1],
    #         running_average(episode_number_of_steps, n_ep_running_average)[-1]))
    
    # # save agent_target
    # network_name = "network_1" + suffix + ".pt"
    # print("network saved as " + network_name)
    # torch.save(agent_target, network_name)

    # result_dict["episode_reward_list"] = episode_reward_list
    # result_dict["episode_number_of_steps"] = episode_number_of_steps
    # return result_dict

result_dict = DQN_learn(env)
episode_reward_list = result_dict["episode_reward_list"] 
episode_number_of_steps = result_dict["episode_number_of_steps"]

# for i in EPISODES:
#     # Reset enviroment data and initialize variables
#     done = False
#     state = env.reset()
#     total_episode_reward = 0.
#     t = 0
#     while not done:
#         # Take a random action
#         action = agent.forward(state)

#         # Get next state and reward.  The done variable
#         # will be True if you reached the goal position,
#         # False otherwise
#         next_state, reward, done, _ = env.step(action)

#         # Update episode reward
#         total_episode_reward += reward

#         # Update state for next iteration
#         state = next_state
#         t+= 1

#     # Append episode reward and total number of steps
#     episode_reward_list.append(total_episode_reward)
#     episode_number_of_steps.append(t)

#     # Close environment
#     env.close()

#     # Updates the tqdm update bar with fresh information
#     # (episode number, total reward of the last episode, total number of Steps
#     # of the last episode, average reward, average number of steps)
#     EPISODES.set_description(
#         "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
#         i, total_episode_reward, t,
#         running_average(episode_reward_list, n_ep_running_average)[-1],
#         running_average(episode_number_of_steps, n_ep_running_average)[-1]))


# possible that length is smaller than N_episodes
N_episodes = len(episode_reward_list)
# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes, gamma: {0}'.format(gamma))
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes, gamma: {0}'.format(gamma))
ax[1].legend()
ax[1].grid(alpha=0.3)
saveName = "tempPlot_" + suffix + ".png"
print("plot saved as " + saveName)
plt.savefig(saveName) 
plt.show()