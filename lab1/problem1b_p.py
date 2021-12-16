# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518 
from matplotlib.pyplot import savefig
from numpy.lib.npyio import save
from minotaurMaze import Maze, dynamic_programming, animate_solution
import numpy as np

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

env = Maze(maze)

# Finite horizon
horizon = 20
# Solve the MDP problem with dynamic programming
V, policy= dynamic_programming(env,horizon)
# Simulate the shortest path starting from position A
method = 'DynProg';
start  = (0,0,6,5);
path = env.simulate(start, policy, method);
suffix = "T_{0}_".format(horizon)
saveName = "maze_proposedReward" + suffix + ".gif"
animate_solution(maze, path, env, createGIF=True, saveFigName=saveName)
print("maze saved at : " + saveName)
print(path)
