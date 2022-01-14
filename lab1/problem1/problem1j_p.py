# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518 
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


start  = (0,0,6,5,0);

epsilon = 0.001
episodes = 50000


# method = 'QLearn'

# alpha = 2./3

# episodes = 50000

# suffix = "_inst_reward_"
# nextMovePolicy = "greedy"

# poisoned = "poisoned" # + "_newStateVisits"

# suffix += nextMovePolicy + "_" + poisoned
# # # save q func, init from scratch, epsilon = const, do not clear visit count to zero after each episode
# # V, policy, iteration_counter, V_over_episodes = qLearning(env, gamma, lambda x: epsilon, lambda x : 1 / np.float_power(x, alpha), episodes,nextMovePolicy,saveQFunc=True, initQFunc="FromScratch", visitCountClearEachEpisode=False)
# # suffix += "_clearVisit_scratch"
# # # save q func, init from scratch, epsilon = const
# # V, policy, iteration_counter, V_over_episodes = qLearning(env, gamma, lambda x: epsilon, lambda x : 1 / np.float_power(x, alpha), episodes,nextMovePolicy,saveQFunc=True, initQFunc="FromScratch")


# # init from previou
# # V, policy, iteration_counter, V_over_episodes = qLearning(env, gamma, lambda x: epsilon, lambda x : 1 / np.float_power(x, alpha), episodes,nextMovePolicy,saveQFunc=True, initQFunc="FromPrevious",visitCountClearEachEpisode=True)
# suffix += "_readPrev_clearVisit"


# # read policy
# policy = np.loadtxt("qLearn_policy" + suffix + ".txt")

# exited_vector = []
# averaged_over = 100
# #lets collect the simulated probability over averaged_over runs and compute mean and std of this
# for run_no in range(averaged_over):

#     #number of exited
#     n_exited = 0
#     #now lets simulate 10000 times
#     for iteration in range(10000):
#         #prob = 29/30 = 1-1/mean
#         path = env.simulate(start, policy, method, prob=gamma)
#         if (env.isExit(path[-1])):
#             n_exited += 1

#     exited_vector.append(n_exited/10000)
# average_exits = np.mean(exited_vector)
# std_exits = np.std(exited_vector)
# print("Average exits over " + str(averaged_over)+" runs : ", average_exits)
# print("Std exits over " + str(averaged_over)+" runs : ", std_exits)


# sarsa

start  = (0,0,6,5,0);

epsilon = 0.001
episodes = 50000


method = 'QLearn'

alpha = 2./3

episodes = 50000

# init from previou
# V, policy, iteration_counter, V_over_episodes = qLearning(env, gamma, lambda x: epsilon, lambda x : 1 / np.float_power(x, alpha), episodes,nextMovePolicy,saveQFunc=True, initQFunc="FromPrevious",visitCountClearEachEpisode=True)


# read policy

# q learn

policy = np.loadtxt("qLearn_policy.txt")
# sarsa
policy = np.loadtxt("sarsa_policy0_200.txt")

exited_vector = []
averaged_over = 100
#lets collect the simulated probability over averaged_over runs and compute mean and std of this
for run_no in range(averaged_over):

    #number of exited
    n_exited = 0
    #now lets simulate 10000 times
    for iteration in range(10000):
        #prob = 29/30 = 1-1/mean
        path = env.simulate(start, policy, method, prob=gamma)
        if (env.isExit(path[-1])):
            n_exited += 1

    exited_vector.append(n_exited/10000)
average_exits = np.mean(exited_vector)
std_exits = np.std(exited_vector)
print("Average exits over " + str(averaged_over)+" runs : ", average_exits)
print("Std exits over " + str(averaged_over)+" runs : ", std_exits)





