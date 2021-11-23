import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    #the avoid getting eaten at each cost
    EATEN_REWARD = -10000


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;

        #so state (i,j,k,l) corresponds to the agent being at (i, j) and minotaur is at (k, l)
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i, j] != 1:
                            states[s] = (i,j,k,l)
                            map[(i, j, k, l)] = s
                            s+=1

        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        new_row_player = self.states[state][0] + self.actions[action][0];
        new_col_player = self.states[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (new_row_player == -1) or (new_row_player == self.maze.shape[0]) or \
                              (new_col_player == -1) or (new_col_player == self.maze.shape[1]) or \
                              (self.maze[new_row_player,new_col_player] == 1);

        # Based on the impossiblity check return the next state.

        if hitting_maze_walls:
            #just return same original position
            return (self.states[state][0], self.states[state][1]);
        else:
            #return the new position
            return (new_row_player, new_col_player);

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):

                #minotaur movement
                row_minotaur = self.states[s][2]
                col_minotaur = self.states[s][3]
                #append all possible directions that the minotaur can walk towards
                list_of_possible_actions_minotaur = []
                for action_minotaur in range(1, self.n_actions):
                    if (row_minotaur + self.actions[action_minotaur][0]) == -1 or (row_minotaur + self.actions[action_minotaur][0] == self.maze.shape[0]) or \
                        (col_minotaur + self.actions[action_minotaur][1]) == -1 or (col_minotaur + self.actions[action_minotaur][1] == self.maze.shape[1]):
                        pass
                    else:
                        list_of_possible_actions_minotaur.append(action_minotaur)
                #the number of possible directions that the minotaur currently can walk towards
                number_of_actions = len(list_of_possible_actions_minotaur)

                #compute the new position of the agent taken into account the state and action
                next_s = self.__move(s,a);
                #go through the positions of how the minotaur moves
                for minotaur_action in list_of_possible_actions_minotaur:
                    #the new state (with both agent and minotaur moving)
                    new_s = self.map[next_s[0], next_s[1], row_minotaur + self.actions[minotaur_action][0], col_minotaur + self.actions[minotaur_action][1]]
                    #transition probabilities are uniform of number of possible directions that the minotaur can walk towards
                    transition_probabilities[new_s, s, a] = 1/number_of_actions;
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    #minotaur movement (same as in __transitions)
                    row_minotaur = self.states[s][2]
                    col_minotaur = self.states[s][3]
                    list_of_possible_actions_minotaur = []
                    for action_minotaur in range(1, self.n_actions):
                        if (row_minotaur + self.actions[action_minotaur][0]) == -1 or (row_minotaur + self.actions[action_minotaur][0] == self.maze.shape[0]) or \
                            (col_minotaur + self.actions[action_minotaur][1]) == -1 or (col_minotaur + self.actions[action_minotaur][1] == self.maze.shape[1]):
                            pass
                        else:
                            list_of_possible_actions_minotaur.append(action_minotaur)
                    number_of_actions = len(list_of_possible_actions_minotaur)

                    next_s = self.__move(s,a);


                    for minotaur_action in list_of_possible_actions_minotaur:
                        new_s = self.map[next_s[0], next_s[1], row_minotaur + self.actions[minotaur_action][0], col_minotaur + self.actions[minotaur_action][1]]

                        #rewards for each state
                        #HERE im unsure, this might be wrong. maybe it shouldnt be looped over. i will check this when i have time maybe on friday

                        # Rewrd for hitting a wall
                        if self.states[s][0:2] == self.states[new_s][0:2] and a != self.STAY:
                            rewards[s,a] = self.IMPOSSIBLE_REWARD;
                        # Reward for reaching the exit
                        elif self.states[s][0:2] == self.states[new_s][0:2] and self.maze[self.states[new_s][0:2]] == 2:
                            rewards[s,a] = self.GOAL_REWARD;

                        #reward for being in the same position as the minotaur (same for all actions)
                        elif self.states[s][0:2] == self.states[s][2:4]:
                            rewards[s,a] = self.EATEN_REWARD
                        #reward for going to the position of where the minotaur currently is
                        #elif self.states[new_s][0:2] == self.states[s][2:4]:
                        #    rewards[s, a] = self.EATEN_REWARD
                        # Reward for taking a step to an empty cell that is not the exit
                        else:
                            rewards[s,a] = self.STEP_REWARD;

                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s]]<0:
                        row, col = self.states[next_s];
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s,a];
                        # With probability 0.5 the reward is
                        r2 = rewards[s,a];
                        # The average reward
                        rewards[s,a] = 0.5*r1 + 0.5*r2;
        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                     next_s = self.__move(s,a);
                     i,j = self.states[next_s];
                     # Simply put the reward as the weights o the next state.
                     rewards[s,a] = weights[i][j];

        return rewards;

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        path_minotaur = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start[0:2]);
            #add the starting position of the minotaur too
            path_minotaur.append(start[2:4])
            while t < horizon-1:
                #minotaur movement, same as in __transitions
                row_minotaur = self.states[s][2]
                col_minotaur = self.states[s][3]
                list_of_possible_actions_minotaur = []
                for action_minotaur in range(1, self.n_actions):
                    if (row_minotaur + self.actions[action_minotaur][0]) == -1 or (row_minotaur + self.actions[action_minotaur][0] == self.maze.shape[0]) or \
                        (col_minotaur + self.actions[action_minotaur][1]) == -1 or (col_minotaur + self.actions[action_minotaur][1] == self.maze.shape[1]):
                        pass
                    else:
                        list_of_possible_actions_minotaur.append(action_minotaur)
                number_of_actions = len(list_of_possible_actions_minotaur)

                random_action_minotaur = np.random.choice(list_of_possible_actions_minotaur)
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s,t]);

                new_s = self.map[next_s[0], next_s[1], row_minotaur + self.actions[random_action_minotaur][0], col_minotaur + self.actions[random_action_minotaur][1]]
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[new_s][0:2])
                path_minotaur.append(self.states[new_s][2:4])
                # Update time and state for next iteration
                t +=1;
                s = new_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        return path, path_minotaur


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)




def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);
    plt.show()
def animate_solution(maze, path, path_minotaur):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):
        if i ==0:
            grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(path[i])].get_text().set_text('Player')
            grid.get_celld()[(path_minotaur[i])].set_facecolor(LIGHT_PURPLE)
            grid.get_celld()[(path_minotaur[i])].get_text().set_text('MINOTAUR')
        if i > 0:
            grid.get_celld()[(path[i-1])].set_facecolor(col_map[maze[path[i-1]]])
            grid.get_celld()[(path[i-1])].get_text().set_text('')

            grid.get_celld()[(path_minotaur[i-1])].set_facecolor(col_map[maze[path_minotaur[i-1]]])
            grid.get_celld()[(path_minotaur[i-1])].get_text().set_text('')

            grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(path[i])].get_text().set_text('Player')
            grid.get_celld()[(path_minotaur[i])].set_facecolor(LIGHT_PURPLE)
            grid.get_celld()[(path_minotaur[i])].get_text().set_text('MINOTAUR')
            if path[i] == path[i-1]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i])].get_text().set_text('Player is out')



        #display.display(fig)
        plt.pause(1)
    plt.show()
        #display.clear_output(wait=True)
        #time.sleep(1)
