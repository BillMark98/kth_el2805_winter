# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518 
from minotaurMaze import Maze, value_iteration, animate_solution
import numpy as np
import matplotlib.pyplot as plt


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

# first try value iteration
V, policy = value_iteration(env, gamma, epsilon)
method = 'ValIter';
start  = (0,0,6,5,0);
path = env.simulate(start, policy, method, prob = gamma);    
# print(path)
animate_solution(maze, path, env,saveFigName = "mazeValIter_keyPicking.gif")

# output the V and policy
np.savetxt("valueIter_V.txt", V, fmt = "%5.4f")
np.savetxt("valIter_policy.txt", policy, fmt = "%5d")



# q-learning