# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518
#  
import os
from minotaurMaze import Maze, qLearning, animate_solution, change2FileDir
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

env = Maze(maze,keyPicking=True)    

# simulate qLearning
life_expectancy = 50
gamma = 1 - 1/life_expectancy
epsilon = 0.01

alpha = 2./3

episodes = 50000

V, policy, iteration_counter, V_over_episodes = qLearning(env, gamma, epsilon, lambda x : 1 / np.float_power(x, alpha), episodes)
method = 'QLearn';
start  = (0,0,6,5,0);
path = env.simulate(start, policy, method, prob = gamma);    
# print(path)
animate_solution(maze, path, env,saveFigName = "mazeQLearn.gif")    
np.savetxt("qLearn_V.txt", V, fmt = "%5.4f")
np.savetxt("qLearn_policy.txt", policy, fmt = "%5d")

# plot the value function of the first state over time
startNum = env.map[start]
plt.plot(list(range(episodes)), V_over_episodes[startNum,:])
plt.xlabel("episodes")
plt.ylabel("value function")
plt.title("value function over episode")
plt.savefig("v_over_eps_qLearn.png")
plt.show()