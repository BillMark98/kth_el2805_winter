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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from torch import optim
from tqdm import trange
from DDPG_agent import RandomAgent, DDPG_agent
# from lab.lab2.problem2.DDPG_agent import DDPG_agent

import os

def change2FileDir():
    """ change working directory to the current file directory"""
    # change to the file position
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

change2FileDir()


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

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
T_E = 400
N_episodes = T_E               # Number of episodes to run for training
discount_factor = 0.95         # Value of gamma
n_ep_running_average = 50      # Running average of 50 episodes
m = len(env.action_space.high) # dimensionality of the action

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []


# parameters
gamma = 0.99
L = 30000
tau = 1e-3
N = 64
d = 2
mu = 0.15
sigma = 0.2
actor_learning_rate = 5e-5
critic_learning_rate = 5e-4
# actor_neuron_layers = 3
# actor_neuronNums = [400,200,200] 

actor_neuron_layers = 2
actor_neuronNums = [400,200] 

# critic_neuron_layers = 3
# critic_neuronNums = [400,200,200] 

critic_neuron_layers = 2
critic_neuronNums = [400,200] 

clipping_value = 1.

n_actions = 2

# # Agent initialization
# agent = RandomAgent(m)

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

# suffix = "Q_nn_{0}_{1}_{2}_pi_nn_{3}_{4}_{5}".format(actor_neuronNums[0],actor_neuronNums[1],actor_neuronNums[2],critic_neuronNums[0],critic_neuronNums[1],critic_neuronNums[2])
suffix = "Q_nn_{0}_{1}_pi_nn_{2}_{3}".format(actor_neuronNums[0],actor_neuronNums[1],critic_neuronNums[0],critic_neuronNums[1])
suffix += "Te_400"


def DDPG_learn(env,
    discount_factor = gamma, buffer_size = L,  train_batch_size = N,
    episodes = EPISODES, d=d, actor_learning_rate = actor_learning_rate, critic_learning_rate=critic_learning_rate, 
    actor_neuron_layers = actor_neuron_layers, actor_neuronNums = actor_neuronNums, 
    critic_neuron_layers = critic_neuron_layers, critic_neuronNums = critic_neuronNums, 
    clipping_value = clipping_value,
    exp_cer = True, premature_stop = True, threshold = 140, optimal_len = 50):

    ''' DDPG learning algorithms
    
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

    -----
    return
    ------

    dictionary: 
    {
        "episode_reward_list": []
        "episode_number_of_steps" : []
    }
    '''


    agent = DDPG_agent(n_actions, discount_factor = discount_factor, buffer_size = buffer_size, 
    train_batch_size = train_batch_size, episodes = N_episodes, 
     actor_learning_rate = actor_learning_rate, 
    critic_learning_rate=critic_learning_rate,
    n_inputs = 8, actor_neuron_layers = actor_neuron_layers, actor_neuronNums= actor_neuronNums, 
    critic_neuron_layers = critic_neuron_layers, critic_neuronNums = critic_neuronNums, 
    gradient_clip= clipping_value, gradient_clip_max=clipping_value, premature_stop = premature_stop, threshold = threshold, optimal_len = optimal_len)

    # initialize experience buffer
    agent.init_ExperienceBuffer(env, updateLen = 50)
    # episode reward and total number of steps lists
    episode_reward_list = []
    episode_number_of_steps = []
    result_dict = {}
    optimal_count_len = 0
    begin_optimal = False
    for episode in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        # reset agent
        agent.reset(episode)
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
            next_state, reward, done, _ = env.step(action)
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

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        # set terminal size smaller to fully print result
        running_average_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
        if (not begin_optimal and (running_average_reward > threshold)):
            begin_optimal = True
            optimal_count_len = 0
        elif (begin_optimal and (running_average_reward > threshold)):
            optimal_count_len += 1
            if (optimal_count_len > optimal_len):
                print("optimal len hit, exit prematurely")
                break
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            episode, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))
    
    # save agent
    Q_network_name = "Q_network_1" + suffix + ".pt"
    print("network saved as " + Q_network_name)
    agent.save_main_Q_nn(Q_network_name)

    pi_network_name = "pi_network_1" + suffix + ".pt"
    print("pi network saved as " + pi_network_name)
    agent.save_main_pi_nn(pi_network_name)
    # torch.save(agent_target, network_name)

    result_dict["episode_reward_list"] = episode_reward_list
    result_dict["episode_number_of_steps"] = episode_number_of_steps
    return result_dict




result_dict = DDPG_learn(env)
episode_reward_list = result_dict["episode_reward_list"] 
episode_number_of_steps = result_dict["episode_number_of_steps"]

# for i in EPISODES:
#     # Reset enviroment data
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

#     # Append episode reward
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


# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
saveName = "tempPlot_" + suffix + ".png"
print("plot saved as " + saveName)
plt.savefig(saveName) 

plt.show()
