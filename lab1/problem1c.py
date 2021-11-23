import maze as mz
import numpy as np
import matplotlib.pyplot as plt

maze = np.array([[0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0, 1, 1, 1],
                 [0, 0, 1, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 1, 1, 0],
                 [0, 0, 0, 0, 1, 2, 0, 0]])

#0 = free space, 1 = obstacle, 2 = goal
env = mz.Maze(maze)
T = 30
#currently only dynoamic_programming works (not value iteration)
V, policy = mz.dynamic_programming(env, T)
method = 'DynProg'

sumvector = [0.]
Tvector = [0]
print(np.shape(env.transition_probabilities[:, :, 0]))
print(np.shape(policy))
start = np.zeros((env.n_states, 1))
start[53] = 1.
trans_prob = start
for i in range(1, T+1):
    print("For T = ", str(i) + ": ")
    new_state = np.zeros((env.n_states, 1))
    for possible_state in range(0, env.n_states):
        if trans_prob[possible_state] ==0.:
            pass
        else:
            action_done = policy[possible_state][i]
            #onehot state
            temp = np.zeros((env.n_states, 1))
            temp[possible_state] = 1.
            new_state += trans_prob[possible_state]* np.matmul(env.transition_probabilities[:,:,int(action_done)], temp)

    trans_prob = new_state
    print("Total probability: ", np.sum(trans_prob))
    sum = 0
    #2072-2127 is all the "winning states"
    for winning_state in range(2072, 2128):
        #print(str(winning_state) + ": ", trans_prob[winning_state])
        sum+=trans_prob[winning_state][0]
    sumvector.append(sum)
    Tvector.append(i)
    #2072 - 2127 = winning states
    print("TOTAL SUM: ", sum)
    print("---------------------------------")

plt.plot(Tvector, sumvector)
plt.title("Probability of reaching (6, 5) vs horizon (T)")
plt.xlabel("T")
plt.ylabel("Probability of escaping")
plt.show()

#path, path_minotaur = env.simulate(start, policy, method)

#mz.animate_solution(maze, path, path_minotaur)
