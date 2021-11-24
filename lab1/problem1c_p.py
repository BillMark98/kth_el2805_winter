from minotaurMaze import Maze, dynamic_programming, animate_solution
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
# Finite horizon
T = 30
# Solve the MDP problem with dynamic programming
# Simulate the shortest path starting from position A
method = 'DynProg';
start  = (0,0,6,5);
start_s = env.map[start]

exit_probability = [0]
time_vector = [0]

#go through T
for horizon_time in range(1, T+1):
    #initialization
    start_vector = np.zeros((env.n_states, 1))
    start_vector[start_s] = 1.
    current_state = start_vector
    V, policy= dynamic_programming(env,horizon_time)
    print("For T = ", str(horizon_time) + ": ")
    for t in range(1, horizon_time+1):
        new_state = np.zeros((env.n_states, 1))
        #go through all possible states (basically we loop through current_state)
        for state in range(env.n_states):
            #if the probability of being in this state is 0, dont bother
            #to calculate (saves some time, we dont need this condition)
            if current_state[state]!=0:
                #the agent will perform the action according to the policy
                action_done = policy[state][t]
                #create one hot encoding of this state to add to new_state
                state_vector = np.zeros((env.n_states, 1))
                state_vector[state] = 1.

                #add to new state. current_state[state] is the probability of being
                #in state state
                new_state += current_state[state]*np.matmul(env.transition_probabilities[:, :, int(action_done)], state_vector)

        #after all states have been added to new state we update current_state
        current_state =new_state
        #sanity check

    print("Total probability: ", np.sum(current_state))
    #print the probability of exiting (reaches goal sucessfully that is the value of the first state)
    exit_probability.append(current_state[0])
    time_vector.append(horizon_time)

print(exit_probability)
plt.plot(time_vector, exit_probability)
plt.title("Exit probability vs horizon T")
plt.xlabel("T")
plt.ylabel("Exit probability")
plt.savefig("exitprobability_problemcplot")
plt.show()
