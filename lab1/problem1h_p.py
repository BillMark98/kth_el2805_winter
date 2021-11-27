# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518
#  
import os
from minotaurMaze import Maze, qLearning, animate_solution, change2FileDir
import numpy as np
import matplotlib.pyplot as plt

change2FileDir()

# key Picking
life_expectancy = 50
gamma = 1 - 1/life_expectancy

maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 4],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])    

env = Maze(maze,keyPicking=True, probability_to_survive=gamma)    

## simple simulation
# simulate qLearning

epsilon = 0.01

alpha = 2./3

episodes = 50000

suffix = "_inst_reward_"
# nextMovePolicy = "_fixed"
nextMovePolicy = "fixed"

poisoned = "poisoned"

suffix += nextMovePolicy + "_" + poisoned
V, policy, iteration_counter, V_over_episodes = qLearning(env, gamma, epsilon, lambda x : 1 / np.float_power(x, alpha), episodes,nextMovePolicy)
method = 'QLearn';
start  = (0,0,6,5,0);
path = env.simulate(start, policy, method, prob = gamma);    
# print(path)
animate_solution(maze, path, env,saveFigName = "mazeQLearn" + suffix + ".gif")    
np.savetxt("qLearn_V" + suffix + ".txt", V, fmt = "%5.4f")
np.savetxt("qLearn_policy" + suffix + ".txt", policy, fmt = "%5d")

# plot the value function of the first state over time
startNum = env.map[start]
plt.figure()
plt.plot(list(range(episodes)), V_over_episodes[startNum,:])
plt.xlabel("episodes")
plt.ylabel("value function")
plt.title("value function over episode")
plt.savefig("v_over_eps_qLearn" + suffix + ".png")
plt.show()

# # 2)

# epsilons = [1, 0.001]
# labels = [str(epsilon) for epsilon in epsilons]
# # simulate qLearning
# life_expectancy = 50
# gamma = 1 - 1/life_expectancy

# alpha = 2./3

# episodes = 50000

# v_start_episodes = np.zeros((len(epsilons), episodes))

# index = 0
# for epsilon in epsilons:
    
#     V, policy, iteration_counter, V_over_episodes = qLearning(env, gamma, epsilon, lambda x : 1 / np.float_power(x, alpha), episodes)
#     method = 'QLearn';
#     start  = (0,0,6,5,0);
#     path = env.simulate(start, policy, method, prob = gamma);    
#     # print(path)
#     # build suffix
#     suffix = "{:.3f}".format(epsilon)
#     suffix = suffix.replace(".", "_")
#     animate_solution(maze, path, env,saveFigName = "mazeQLearn" + suffix + ".gif")    

#     np.savetxt("qLearn_V" + suffix + ".txt", V, fmt = "%5.4f")
#     np.savetxt("qLearn_policy" + suffix + ".txt", policy, fmt = "%5d")
#     startNum = env.map[start]
#     v_start_episodes[index, :] = V_over_episodes[startNum,:]
#     index += 1
# # plot the value function of the first state over time

# plt.figure()
# for index in range(len(epsilons)):
#     plt.plot(list(range(episodes)), v_start_episodes[index,:], label = labels[index])
# plt.legend()
# plt.xlabel("episodes")
# plt.ylabel("value function")
# plt.title("value function over episode")
# plt.savefig("v_over_eps_qLearn.png")
# plt.show()

# 3)

# alphas = [0.6, 0.9]
# labels = [str(alpha) for alpha in alphas]
# # simulate qLearning
# life_expectancy = 50
# gamma = 1 - 1/life_expectancy

# epsilon = 0.001
# episodes = 50000

# v_start_episodes = np.zeros((len(alphas), episodes))

# index = 0
# for alpha in alphas:
    
#     V, policy, iteration_counter, V_over_episodes = qLearning(env, gamma, epsilon, lambda x : 1 / np.float_power(x, alpha), episodes)
#     method = 'QLearn';
#     start  = (0,0,6,5,0);
#     path = env.simulate(start, policy, method, prob = gamma);    
#     # print(path)
#     # build suffix
#     suffix = "alpha_{:.3f}".format(alpha)
#     suffix = suffix.replace(".", "_")
#     animate_solution(maze, path, env,saveFigName = "mazeQLearn" + suffix + ".gif")    

#     np.savetxt("qLearn_V" + suffix + ".txt", V, fmt = "%5.4f")
#     np.savetxt("qLearn_policy" + suffix + ".txt", policy, fmt = "%5d")
#     startNum = env.map[start]
#     v_start_episodes[index, :] = V_over_episodes[startNum,:]
#     index += 1
# # plot the value function of the first state over time

# plt.figure()
# for index in range(len(alphas)):
#     plt.plot(list(range(episodes)), v_start_episodes[index,:], label = labels[index])
# plt.legend()
# plt.xlabel("episodes")
# plt.ylabel("value function")
# plt.title("value function over episode, varying alpha")
# plt.savefig("v_over_eps_qLearn_alpha.png")
# plt.show()