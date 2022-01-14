# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518
 
import os
from minotaurMaze import Maze, sarsa, animate_solution, change2FileDir, printText
import numpy as np
import matplotlib.pyplot as plt

change2FileDir()

# key Picking

maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 4],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])    

# simulate sarsa
life_expectancy = 50
gamma = 1 - 1/life_expectancy

# env = Maze(maze,keyPicking=True,greedyGoal=True, probability_to_survive=gamma, scaleReward=True)    
env = Maze(maze,keyPicking=True,greedyGoal=True, probability_to_survive=gamma, scaleReward=False)    



epsilon = 0.2

alpha = 2./3

episodes = 50000

# # print("start index : {0}".format(env.map[(0,0,6,5,0)]))
# suffix = "_inst_reward_"
# # nextMovePolicy = "_fixed"
# # nextMovePolicy = "fixed"

# poisoned = "poisoned"

# suffix += poisoned
# suffix += "_fixeps{0:.0e}_".format(epsilon)
# initQFuncMethod = "FromPrevious"
# # initQFuncMethod = "FromScratch"

# suffix += initQFuncMethod
# # suffix += initQFuncMethod + "2"

# delta = 1
# suffix += "delta_{0:.0e}".format(delta)
# # # previous result Q init, clear count
# # V, policy, iteration_counter, V_over_episodes = sarsa(env, gamma, lambda x : epsilon, lambda x : 1 / np.float_power(x, alpha), episodes, initQFunc=initQFuncMethod, visitCountClearEachEpisode=False)
# V, policy, iteration_counter, V_over_episodes = sarsa(env, gamma, lambda x : 1 / np.float_power(x, delta), lambda x : 1 / np.float_power(x, alpha), episodes, initQFunc=initQFuncMethod, visitCountClearEachEpisode=False)



# # suffix += "_NoScaleReward_previous_newReward4"
# # suffix += "_NoScaleReward_oldReward_origMino_previous_correctTrans_accum"
# # suffix += "


# # # # previous result Q init, from qlearning, converging!!!!
# # V, policy, iteration_counter, V_over_episodes = sarsa(env, gamma, lambda x : epsilon, lambda x : 1 / np.float_power(x, alpha), episodes, initQFunc="FromPrevious", QFuncFileNameRead="qFunc_qLearn.txt")

# # from scratch init
# # V, policy, iteration_counter, V_over_episodes = sarsa(env, gamma, lambda x : epsilon, lambda x : 1 / np.float_power(x, alpha), episodes, initQFunc="FromScratch",visitCountClearEachEpisode=False)
# # # suffix += "_NoScaleReward_oldReward_origMino_correctTrans"
# # suffix += "_NoScaleReward_newReward_origMino_correctTrans_AccumCount"


# # const learning rate
# # V, policy, iteration_counter, V_over_episodes = sarsa(env, gamma, lambda x : epsilon, lambda x : 0.8, episodes, initQFunc="FromScratch",visitCountClearEachEpisode=True)


# # suffix += "_NoScaleReward_constLearn"
# # suffix += "_scaleReward"

# # # from scratch init, decreasing epsilon
# # delta = 0.7
# # V, policy, iteration_counter, V_over_episodes = sarsa(env, gamma, lambda x : 1./np.float_power(x, delta), lambda x : 1 / np.float_power(x, alpha), episodes, initQFunc="FromScratch")
# # suffix += "_NoScaleReward_scratch_newReward3_decElon"


# # V, policy, iteration_counter, V_over_episodes = sarsa(env, gamma, epsilon, lambda x : 0.8, episodes)

# method = 'SARSA';
# start  = (0,0,6,5,0);
# # policy = np.loadtxt("sarsa_policy_temp_inst_reward_poisoned_NoScaleReward_previous_newReward.txt")
# # sarsa_2ndRnd_noscale_newReward.txt
# path = env.simulate(start, policy, method, prob = gamma);    
# # print(path)
# animate_solution(maze, path, env,saveFigName = "mazeSARSA_temp" + suffix + ".gif")    
# printText("saved maze : " + "mazeSARSA_temp" + suffix + ".gif")
# np.savetxt("sarsa_V_temp" + suffix + ".txt", V, fmt = "%5.4f")
# printText("saved value function : " + "sarsa_V_temp" + suffix + ".txt")
# np.savetxt("sarsa_policy_temp" + suffix + ".txt", policy, fmt = "%5d")
# printText("saved policy : " + "sarsa_policy_temp" + suffix + ".txt")

# plt.figure()

# # plot the value function of the first state over time
# startNum = env.map[start]
# plt.plot(list(range(episodes)), V_over_episodes[startNum,:])
# plt.xlabel("episodes")
# plt.ylabel("value function")
# plt.title("value function over episode")
# plt.savefig("v_over_eps_sarsa_temp" + suffix + ".png")
# printText("saved value over epsisodes: " + "v_over_eps_sarsa_temp" + suffix + ".png")
# # from value Iter: -48...
# plt.show()


# # 2)

# deltas = [0.6, 0.9]
# labels = [str(delta) for epsilon in deltas]
# # simulate qLearning
# life_expectancy = 50
# gamma = 1 - 1/life_expectancy

# alpha = 2./3

# episodes = 50000

# v_start_episodes = np.zeros((len(epsilons), episodes))

# index = 0
# for epsilon in epsilons:
#     V, policy, iteration_counter, V_over_episodes = sarsa(env, gamma, lambda x : epsilon, lambda x : 1 / np.float_power(x, alpha), episodes, initQFunc="FromScratch", visitCountClearEachEpisode=True)
#     method = 'SARSA';
#     start  = (0,0,6,5,0);
#     path = env.simulate(start, policy, method, prob = gamma);
#     # print(path)
#     # build suffix
#     suffix = "{:.3f}".format(epsilon)
#     suffix = suffix.replace(".", "_")
#     # animate_solution(maze, path, env,saveFigName = "mazesarsa" + suffix + ".gif")

#     np.savetxt("sarsa_V" + suffix + ".txt", V, fmt = "%5.4f")
#     np.savetxt("sarsa_policy" + suffix + ".txt", policy, fmt = "%5d")
#     startNum = env.map[start]
#     v_start_episodes[index, :] = V_over_episodes[startNum,:]
#     index += 1
# # # plot the value function of the first state over time
# plt.figure()
# for index in range(len(epsilons)):
#     plt.plot(list(range(episodes)), v_start_episodes[index,:], label = labels[index])
# plt.legend()
# plt.xlabel("episodes")
# plt.ylabel("value function")
# plt.title("value function over episode")
# plt.savefig("v_over_eps_sarsa_diffElon.png")

# plt.show()


# 3)

deltas = [0.6, 0.9, 1.]
labels = [str(delta) for delta in deltas]
# simulate qLearning
life_expectancy = 50
gamma = 1 - 1/life_expectancy

alpha = 2./3

episodes = 50000

v_start_episodes = np.zeros((len(deltas), episodes))

suffix = "sarsa_"
initQFuncMethod = "FromPrevious"
# initQFuncMethod = "FromScratch"
suffix += initQFuncMethod
index = 0
for delta in deltas:
    V, policy, iteration_counter, V_over_episodes = sarsa(env, gamma, lambda x : 1 / np.float_power(x, delta), lambda x : 1 / np.float_power(x, alpha), episodes, initQFunc=initQFuncMethod, visitCountClearEachEpisode=False, QFuncFileNameWrite="qTemp.txt")
    method = 'SARSA';
    start  = (0,0,6,5,0);
    path = env.simulate(start, policy, method, prob = gamma);
    # print(path)
    # build suffix
    suffix = "{:.3f}".format(epsilon)
    suffix = suffix.replace(".", "_")
    # animate_solution(maze, path, env,saveFigName = "mazesarsa" + suffix + ".gif")

    np.savetxt("sarsa_V" + suffix + ".txt", V, fmt = "%5.4f")
    np.savetxt("sarsa_policy" + suffix + ".txt", policy, fmt = "%5d")
    startNum = env.map[start]
    v_start_episodes[index, :] = V_over_episodes[startNum,:]
    index += 1
# # plot the value function of the first state over time
plt.figure()
for index in range(len(deltas)):
    plt.plot(list(range(episodes)), v_start_episodes[index,:], label = labels[index])
plt.legend()
plt.xlabel("episodes")
plt.ylabel("value function")
plt.title("value function over episode varying delta")
fileName = "v_over_eps_sarsa_diffDelta_" + suffix + ".png"
plt.savefig(fileName)

print("saved over {0}".format(fileName))

plt.show()