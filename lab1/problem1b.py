import maze as mz
import numpy as np

maze = np.array([[0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0, 1, 1, 1],
                 [0, 0, 1, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 1, 1, 0],
                 [0, 0, 0, 0, 1, 2, 0, 0]])

#0 = free space, 1 = obstacle, 2 = goal
env = mz.Maze(maze)
T = 20
#currently only dynoamic_programming works (not value iteration)
V, policy = mz.dynamic_programming(env, T)
np.save("policy1", policy)
np.save("states1", env.states)
np.save("map1", env.map)
method = 'DynProg'

#print(V[53])
print(policy[53])
start = (0, 0, 6, 5)
#start = "isEaten"
print(env.map[start])
#env.map(start) = 53


path, path_minotaur = env.simulate(start, policy, method)

mz.animate_solution(maze, path, path_minotaur)
