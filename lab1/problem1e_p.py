# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518 
from minotaurMaze import Maze, dynamic_programming, animate_solution, value_iteration
import numpy as np
import matplotlib.pyplot as plt

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

start  = (0,0,6,5);
discount_factor = 29/30
epsilon = 0.01
V, policy = value_iteration(env, discount_factor, epsilon)

method = 'ValIter'




exited_vector = []
averaged_over = 10
#lets collect the simulated probability over averaged_over runs and compute mean and std of this
for run_no in range(averaged_over):

    #number of exited
    n_exited = 0
    #now lets simulate 10000 times
    for iteration in range(10000):
        #prob = 29/30 = 1-1/mean
        path = env.simulate(start, policy, method, prob=discount_factor)
        if (env.isExit(path[-1])):
            n_exited += 1

    exited_vector.append(n_exited/10000)
average_exits = np.mean(exited_vector)
std_exits = np.std(exited_vector)
print("Average exits over " + str(averaged_over)+" runs : ", average_exits)
print("Std exits over " + str(averaged_over)+" runs : ", std_exits)
