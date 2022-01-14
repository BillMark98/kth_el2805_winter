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
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#
# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518 
# Load packages
from functools import reduce
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import random
import os
import pickle

def change2FileDir():
    """ change working directory to the current file directory"""
    # change to the file position
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

change2FileDir()
# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 320        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma


# Reward
episode_reward_list = []  # Used to save episodes reward


# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x

def getValuePolicy(w, eta_basis, state):
    """ given state, w, eta_basis, calculate the optimal policy
    
    -----
    return
    ----

        {
            "value" : -1,
            "action" : 0
        }
    """
    global env
    n_actions = env.action_space.n
    m = np.shape(eta_basis)[1]
    maxVal = -np.infty
    opt_act = -1
    value_policy = dict()
    for action in range(n_actions):
        phi_s = [np.cos(np.pi * np.dot(eta_basis[:, index], state)) for index in range(m)]
        current_q = np.dot(w[:, action], phi_s)
        if current_q > maxVal:
            opt_act = action
            maxVal = current_q
    value_policy["action"] = opt_act
    value_policy["value"] = maxVal
    return value_policy

def getValue(w, eta_basis, x_coord, y_coord):
    """ calculate the value"""
    value_policy = getValuePolicy(w, eta_basis, np.array([x_coord, y_coord]))
    return value_policy["value"]

def getValueMesh(w, eta_basis, x_mesh, y_mesh):
    """ calculate the value"""
    value_mesh = np.zeros(np.shape(x_mesh))
    for index_x in range(np.shape(x_mesh)[0]):
        for index_y in range(np.shape(y_mesh)[1]):
            value_mesh[index_x, index_y] = getValue(w, eta_basis, x_mesh[index_x, index_y], y_mesh[index_x, index_y])            
    return value_mesh

def getPolicy(w, eta_basis, x_coord, y_coord):
    """ calculate the policy"""
    value_policy = getValuePolicy(w, eta_basis, np.array([x_coord, y_coord]))
    return value_policy["action"]

def getPolicyMesh(w, eta_basis, x_mesh, y_mesh):
    """ calculate the value"""
    policy_mesh = np.zeros(np.shape(x_mesh))
    for index_x in range(np.shape(x_mesh)[0]):
        for index_y in range(np.shape(y_mesh)[1]):
            policy_mesh[index_x, index_y] = getPolicy(w, eta_basis, x_mesh[index_x, index_y], y_mesh[index_x, index_y])            
    return policy_mesh


def sarsa_lambda(env, eta_basis, var_lambda, learningRateFunc,  \
        epsilonFunc , gamma = discount_factor, episodes = 200, \
             w_initMethod = "AllZero", w_writeFileName = "w_val.txt",\
                  w_readFileName = "w_val.txt", sgdMod = True, sgdMethod = "momentum",\
                      momentum_param = 0.6 , clipElig = True, \
                      clipLow = -5, clipHigh = 5, scaleLearningRate = True, reduceLR = True, target = -130):
    """ implement the sarsa_lambda algorithms
    
    ---
    Parmameters:
    ---

    env : the environment

    eta_basis: the basis used for the fourier expansion
    of the form (2,m), where each is a 2 dimensional vector, each entry is from {0,1,2}

    var_lambda: the lambda appeared in the eligibility calulation

    learningRateFunc:  the learning rate function

    epsilonFunc : the value used for the epsilon-greedy decision
        ensures that each action will be visited with prob >= epsilon / n_actions, where n_actions is the total number of actions available

    gamma: the discound factor, in this problem 1

    episodes: the length of total episodes, default 200

    w_initMethod:
     method used to init the w vector 
        "AllZero" : all to zero
        "FromPrevious" : read from a file with values calculated before, file name is w_readFileName

    w_writeFileName:
    
    the file to write the calculated w value, default "w_val.txt"
    
    w_readFileName :
    the file to read the w value from for the init, default  "w_val.txt"

    sgdMod : indicate whether modify the sgd, default True
    
    sgdMethod = "momentum" or "nesterov"

    momentum_param = 0.6 

    clipElig : boolean indicate to clip the eligibility to [clipLow, clipHigh], default  True
    
    clipLow : default -5
    
    clipHigh: default 5

    scaleLearningRate: boolean indicate whether to scale learning rate for each w_i w.r.t basis eta_i as suggested
    default True

    reduceLR: reduce learning rate by 30% when close to the target
    ---
    Return
    ---

    a dictionary

    {
        "w": m * A dimensional matrix, each w_a  is a m-dimensional vector, corresponds to an action a
        "episode_reward" : an array of episode reward
    }

    """

    n_actions = env.action_space.n  
    m = np.shape(eta_basis)[1]

    w = np.zeros((m,n_actions))
    if w_initMethod == "AllZero":
        pass
    elif w_initMethod == "FromPrevious":
        w = np.loadtxt(w_readFileName)
    else:
        raise Exception("unknown init method for w matrix")

    # eligibility trace vector
    z = np.zeros((m, n_actions))

    # given the state s, calculate the phi function
    def Phi_S(s):
        phi_s = [np.cos(np.pi * np.dot(eta_basis[:, index], s)) for index in range(m)]
        return np.array(phi_s)
    # evaluate the q function

    def Q_evaluate(s,a):
        """ given state s, action a, calculate the q function
    
        """
        # form the phi(s)
        phi_s = Phi_S(s)
        # aaa = np.dot(w[:, a], phi_s)
        
        return np.dot(w[:, a], phi_s)
    epsilon = epsilonFunc(1)
    # help function to get next move
    def nextMoveEpsSoft(s, eps = epsilon):
        """ return next currentState_actionNum with epsilon soft policy"""
        # choose random action if random gives a value less than eps
        if np.random.random() < eps:
            return random.sample(range(n_actions), 1)[0]
        else :
            # choose the greedy

            # create a vector
            max_action_reward = -np.infty
            max_action = 0
            for action in range(n_actions):
                temp_reward = Q_evaluate(s,action)
                if (temp_reward > max_action_reward):
                    max_action_reward = temp_reward
                    max_action = action
            return max_action

    def updateEligibilityTrace(s_t, a_t):
        """ update the eligibility trace """
        nonlocal z, var_lambda, gamma
        for action in range(n_actions):
            old_z_value = z[:,action]
            z[:,action] = gamma * var_lambda * old_z_value
            if (action == a_t):
                z[:, action] += Phi_S(s_t)
        if(clipElig):
            z = np.clip(z, clipLow, clipHigh)
    
    # return dictionary 
    sarsa_response = dict()
    episode_reward_list = [] 
    for episode in range(episodes):
        # Reset enviroment data
        done = False
        state = scale_state_variables(env.reset())
        epsilon = epsilonFunc(episode + 1)
        if (sgdMod):
            # init v
            # v = np.ones((m, n_actions))
            v = np.zeros((m, n_actions))
        total_episode_reward = 0.
        # reset eligibility trace
        z = np.zeros((m, n_actions))
        while not done:
            # Take a random action
            # env.action_space.n tells you the number of actions
            # available
            # action = np.random.randint(0, k)
            action = nextMoveEpsSoft(state, epsilon)
                
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)
            next_state = scale_state_variables(next_state)
            next_action = nextMoveEpsSoft(next_state, epsilon)

            # temporal difference
            temporal_difference = reward + gamma * Q_evaluate(next_state, next_action) - Q_evaluate(state, action)
            # update the z
            updateEligibilityTrace(state, action)
            # update the w
            # first obtain current basic learnign rate alpha
            learning_alpha = learningRateFunc(episode + 1)
            if (reduceLR):
                if ((epsilon >= 50) and running_average(episode_reward_list, 50) >= target):
                    learning_alpha *= 0.3
            alpha_array = [learning_alpha] * m

            if (scaleLearningRate):
                # create scaled alpha diagonal matrix
                for index in range(m):
                    if (np.linalg.norm(eta_basis[:,index]) > 0.01):
                        # norm is not 0
                        alpha_array[index] /= np.linalg.norm(eta_basis[:,index])
            alpha_diag_matrix = np.diag(alpha_array)
            if (sgdMod):
                if (sgdMethod == "momentum"):
                    v = momentum_param * v + temporal_difference * np.dot( alpha_diag_matrix, z)
                    w = w + v
                elif (sgdMethod == "nesterov"):
                    v = momentum_param * v + temporal_difference * np.dot( alpha_diag_matrix, z)
                    w = w + momentum_param * v + temporal_difference * np.dot( alpha_diag_matrix, z)
            else:
                w = w + temporal_difference * np.dot(alpha_diag_matrix, z)

            # # update the z
            # updateEligibilityTrace(state, action)
            # Update episode reward
            total_episode_reward += reward
                
            # Update state for next iteration
            state = next_state

        # Append episode reward
        episode_reward_list.append(total_episode_reward) 

    sarsa_response["episode_reward_list"] = episode_reward_list
    sarsa_response["w"]  = w
    np.savetxt(w_writeFileName, w, fmt="%.10f")
    env.close()
    return sarsa_response

def plotPolicyFunc(w, eta_basis, saveFileName = "policy.png"):
    """ given w, eta_basis, plot value function"""

    s1_arrays = np.arange(0,1,0.01)
    s2_arrays = np.arange(0,1,0.01)

    X_coords, Y_coords = np.meshgrid(s1_arrays, s2_arrays)
    valueFunc = getPolicyMesh(w,eta_basis, X_coords, Y_coords)
    fig= plt.figure()
    ax = plt.axes(projection='3d')
    # ax.contour3D(X_coords, Y_coords, valueFunc, 50, cmap='binary')
    ax.plot_surface(X_coords, Y_coords, valueFunc, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    ax.set_xlabel('s1')
    ax.set_ylabel('s2')
    ax.set_zlabel('policy');
    plt.savefig(saveFileName)
    print("saved at {0}".format(saveFileName))
    plt.show()

def plotRewards(episode_reward_list, titleName = "Total Reward vs Episodes", savefigName = "rewards.png", plotAverage = True):
    """ plot the episode rewards and running average"""
    # Plot Rewards
    N_episodes = len(episode_reward_list)
    plt.plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
    if(plotAverage):
        plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 50), label='Average episode reward')
        plt.legend()

    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title(titleName)
    plt.grid(alpha=0.3)

    plt.savefig(savefigName)
    plt.show()


def simulate(env, policyMethod, policyFunc,episodes):
    """ run a simulation
    
    ----
    Parameters
    ----

    policyFunc : given state (2d vector), will give back action
    """

    episode_reward_list = []
    for episode in range(episodes):
        # Reset enviroment data
        done = False
        state = scale_state_variables(env.reset())
        total_episode_reward = 0.
        while not done:
            # Take a random action
            # env.action_space.n tells you the number of actions
            # available
            if (policyMethod == "random"):
                action = np.random.randint(0, k)
            else:
                action = policyFunc(state)
                
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)
            next_state = scale_state_variables(next_state)

            total_episode_reward += reward
                
            # Update state for next iteration
            state = next_state

        # Append episode reward
        episode_reward_list.append(total_episode_reward) 

    # sarsa_response["episode_reward_list"] = episode_reward_list
    # sarsa_response["w"]  = w

    # plotRewards(episode_reward_list, titleName = "Total reward vs Episodes " + policyMethod)
    # np.savetxt(w_writeFileName, w, fmt="%.10f")
    env.close()
    return episode_reward_list


def plotValueFunc(w, eta_basis,saveFileName="valueFunc.png"):
    """ given w, eta_basis, plot value function"""

    s1_arrays = np.arange(0,1,0.01)
    s2_arrays = np.arange(0,1,0.01)

    X_coords, Y_coords = np.meshgrid(s1_arrays, s2_arrays)
    valueFunc = getValueMesh(w,eta_basis, X_coords, Y_coords)
    fig= plt.figure()
    ax = plt.axes(projection='3d')
    # ax.contour3D(X_coords, Y_coords, valueFunc, 50, cmap='binary')

    ax.plot_surface(X_coords, Y_coords, valueFunc, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    ax.set_xlabel('s1')
    ax.set_ylabel('s2')
    ax.set_zlabel('value_func');
    plt.savefig(saveFileName)
    print("saved at {0}".format(saveFileName))
    plt.show()



def plotRewardOverAlpha(N_episodes, alpha_min, alpha_max, totalNum = 4, titleName = "Average Rewards over alpha", savefigName = "reward_alpha.png"):
    """ plot the average rewards over alpha """

    var_lambda = 0.8
    gamma = 0.99
    delta = 0.9
    alpha_lists = np.linspace(alpha_min, alpha_max, totalNum)
    labels = ["{:.4f}".format(alpha) for alpha in alpha_lists]
    sarsa_response = dict()
    count = 0
    for alpha in alpha_lists:
        sarsa_response = sarsa_lambda(env, eta_basis,var_lambda, lambda x: alpha, lambda x : 1 / np.float_power(x, delta), gamma = gamma, episodes=N_episodes,w_initMethod="AllZero",sgdMod=True)
        episode_reward_list = sarsa_response["episode_reward_list"]
        # plotRewards(episode_reward_list=)

        plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 50), label=labels[count])
        count += 1
        # plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('reward')
    plt.title(titleName)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.savefig(savefigName)
    plt.show()


def plotRewardOverLambda(N_episodes, var_lambda_min, var_lambda_max, totalNum = 4, titleName = "Average Rewards over lambda", savefigName = "reward_lambda.png"):
    """ plot the average rewards over var_lambda """


    alpha = 0.001
    gamma = 0.99
    delta = 0.9
    var_lambda_lists = np.linspace(var_lambda_min, var_lambda_max, totalNum)
    labels = [str(var_lambda) for var_lambda in var_lambda_lists]
    for index, var_lambda in enumerate(var_lambda_lists):
        sarsa_response = sarsa_lambda(env, eta_basis,var_lambda, lambda x: alpha, lambda x : 1 / np.float_power(x, delta), gamma = gamma, episodes=N_episodes,w_initMethod="AllZero",sgdMod=True)
        episode_reward_list = sarsa_response["episode_reward_list"]
        # plotRewards(episode_reward_list=)

        plt.plot([i for i in range(1, N_episodes+1)], running_average(episode_reward_list, 50), label=labels[index])
        # plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('reward')
    plt.title(titleName)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(savefigName)
    plt.show()


# # Training process
# for i in range(N_episodes):
#     # Reset enviroment data
#     done = False
#     state = scale_state_variables(env.reset())
#     total_episode_reward = 0.

#     while not done:
#         # Take a random action
#         # env.action_space.n tells you the number of actions
#         # available
#         action = np.random.randint(0, k)
            
#         # Get next state and reward.  The done variable
#         # will be True if you reached the goal position,
#         # False otherwise
#         next_state, reward, done, _ = env.step(action)
#         next_state = scale_state_variables(next_state)

#         # Update episode reward
#         total_episode_reward += reward
            
#         # Update state for next iteration
#         state = next_state

#     # Append episode reward
#     episode_reward_list.append(total_episode_reward)

#     # Close environment
#     env.close()
    
next_state = [0.22622061, 0.48682418]
next_action = 0
state = [0.22724538, 0.4811313 ]
action = 0

# eta_basis = np.array([[0,0],
# [1,0],
# [0,1],
# [1,1],
# [2,1],
# [1,2]])

# eta_basis = np.array([[0,0],
# [1,0],
# [0,1],
# [1,1]])
eta_basis = np.array([
[1,0],
[0,1],
[1,1]])
eta_basis = np.transpose(eta_basis)

var_lambda = 0.8
alpha = 0.001
epsilon = 0.1
gamma = 0.99
delta = 0.9
sarsa_response = sarsa_lambda(env, eta_basis,var_lambda, lambda x: alpha, lambda x : 1 / np.float_power(x, delta), gamma = gamma, episodes=N_episodes,w_initMethod="AllZero",sgdMod=True)
# sarsa_response = sarsa_lambda(env, eta_basis,var_lambda, lambda x: alpha, lambda x : 1 / np.float_power(x, delta), gamma = gamma, episodes=N_episodes,w_initMethod="FromPrevious",sgdMod=True)

episode_reward_list = sarsa_response["episode_reward_list"]
w = sarsa_response["w"]

# d) 1)plot rewards
plotRewards(episode_reward_list)

# # d) 2)
# plot 3d plot of value function of optimal policy over state space
plotValueFunc(w, eta_basis)

# d) 3)
# plot 3d plot of optimal policy over state space
plotPolicyFunc(w, eta_basis)

# d) 5)
dummy = ""
random_agent_list = simulate(env,"random",dummy, 50)
sarsa_lambda_list = simulate(env, "behavior policy", lambda x : getPolicy(w, eta_basis, x[0],x[1]),50)

saveName = "Random_SARSA_rewards.png"
plt.plot(list(range(1,len(sarsa_lambda_list) + 1)), sarsa_lambda_list, label = 'sarsa_lambda')
plt.plot(list(range(1,len(random_agent_list) + 1)), random_agent_list, label = 'random')
plt.title("Random vs SARSA_lambda")
plt.ylabel("reward")
plt.xlabel("episode")
plt.legend()
print("plot saved as " + saveName)
plt.savefig(saveName)
plt.show()

# e)

plotRewardOverAlpha(N_episodes=320, alpha_min=0.001, alpha_max=0.01)
plotRewardOverLambda(N_episodes=320, var_lambda_min=0.5, var_lambda_max=0.9)


# f)

data = {'W' : np.transpose(w),
'N' : np.transpose(eta_basis)}

with open("weights_temp.pkl","wb") as f:
    pickle.dump(data, f)
