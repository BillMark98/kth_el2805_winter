from minotaurMaze import Maze, dynamic_programming, animate_solution
import numpy as np

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
animate_solution(maze, path, env, createGIF=False)
print(path)
