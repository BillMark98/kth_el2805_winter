# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518 
import os
from minotaurMaze import Maze, value_iteration, animate_solution, change2FileDir, printText
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

env = Maze(maze,keyPicking=True,scaleReward=False)    

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

# suffix = "_NoScaleReward_randMinoMove"
# suffix = "_NoScaleReward_correctMove_NewReward4"
# suffix = "_NoScaleReward_correctMove_OldReward"
suffix = "_NoScaleReward_correctMove_proposedReward"



animate_solution(maze, path, env,saveFigName = "mazeValIter_keyPicking_mac" + suffix + ".gif")
printText("saved gif: " + "mazeValIter_keyPicking_mac" + suffix + ".gif")
# output the V and policy
np.savetxt("valueIter_V_mac" + suffix + ".txt", V, fmt = "%5.4f")
printText("saved value function : " + "valueIter_V_mac" + suffix + ".txt")
np.savetxt("valIter_policy_mac" + suffix + ".txt", policy, fmt = "%5d")
printText("saved policy : " + "valIter_policy_mac" + suffix + ".txt")

print("value function of start pos")
print(V[env.map[start]])
# 0.5679761759500594
# -97.10525298752762
# old -48.552626493763846