from minotaurMaze import value_iteration
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

startPos = (0,0,6,5)
method = "ValIter"
TTL = 30

life_expectancy = 30
gamma = 1 - 1/life_expectancy
epsilon = 0.01

timeOfGames = 10000

successCount = 0
# calculate the policy
V, policy = value_iteration(env, gamma, epsilon)
    
for episode in range(timeOfGames):
    # simulate the value iteration
    path = env.simulate(startPos, policy, method, TTL = TTL)
    # test if exit
    if (env.isExit(path[-1])):
        successCount += 1

print("probability of exiting the maze : {0}".format(successCount / timeOfGames))