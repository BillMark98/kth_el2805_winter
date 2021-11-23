import numpy as np
import matplotlib.pyplot as plt
import time
import operator
import random
from IPython import display
from matplotlib.animation import FuncAnimation

import os



# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED, 3: LIGHT_PURPLE};

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
    EATEN_REWARD = -1000

    # special state enumeration
    STATE_EXIT = 0
    STATE_EATEN = 1


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.

        maze convention:
            0 : free square to move
            1 : wall
            2 : exit
            3 : minotaur
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);
    def isEaten(self, pos_state):
        """ given position state pos_state, test if sueseth is eaten
            pos_state : four element tuple
            : return boolean
        """
        return (pos_state[0] == pos_state[2]) and (pos_state[1] == pos_state[3])

    def isExit(self, pos_state):
        """ given position state, test if is at the exit
            : return boolean
        """
        return self.maze[pos_state[0], pos_state[1]] == 2

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
        # stateNum count from 2, the first two reserved
        s = 2;
        states[0] = 0;
        states[1] = 1;
        # at most S * S states,
        # S is the number of squares of the maze
        # pos_state is of the form (i,j,mx,my), where (i,j) is the position of the explorer theseus (mx,my) pos of minotaur
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i,j] != 1:
                    for mx in range(self.maze.shape[0]):
                        for my in range(self.maze.shape[1]):
                            if not (self.isEaten((i,j,mx,my)) or self.isExit((i,j,mx,my))):
                                states[s] = (i,j,mx,my);
                                map[(i,j, mx,my)] = s;
                                s += 1;
        return states, map

    def __getStateNum(self, pos_state):
        """ given position state (4-tuple) get the state number

            : return integer
        """

        if self.isExit(pos_state):
            stateNum = self.STATE_EXIT
        elif self.isEaten(pos_state):
            stateNum = self.STATE_EATEN
        else:
            # normal state use map
            stateNum = self.map[pos_state]
        return stateNum
    def __coordinateAddition(self, coord_1, coord_2):
        """ given coord_1, and coord2  two tuples, add componentwise, return a tuple
        """
        return tuple(map(operator.add, coord_1, coord_2))

    def __isHittingWall(self, pos_state):
        """ given a tuple, test if it hits a wall
            : return a boolean
        """
        row = pos_state[0]
        col = pos_state[1]
        return (row == -1) or (row == self.maze.shape[0]) or \
                                (col == -1) or (col == self.maze.shape[1]) or \
                                (self.maze[row,col] == 1);

    def __randomMove(self, singlePoint_pos):
        """ randomMove, use to simulate the move of minotaur.
            if no constraint, choose one of the four randomly, otherwise, choose the possible moves uniform randomly

            ---
            Parameters
            ---
            singlePoint_pos : a 2-element tuple, the usual coordinate of a point

            ---
            return
            ---
            return a list of possible states
        """
        row = singlePoint_pos[0]
        col = singlePoint_pos[1]
        possibleActions = [self.MOVE_LEFT,self.MOVE_RIGHT, self.MOVE_UP, self.MOVE_DOWN]

        if row == 0:
            possibleActions.remove(self.MOVE_UP)
        elif row == self.maze.shape[0] - 1 :
            possibleActions.remove(self.MOVE_DOWN)

        if col == 0:
            possibleActions.remove(self.MOVE_LEFT)
        elif col == self.maze.shape[1] - 1 :
            possibleActions.remove(self.MOVE_RIGHT)

        # get the actionMoves
        actionMoves = [self.actions[actionName] for actionName in possibleActions]

        possibleNextStates = [self.__coordinateAddition(singlePoint_pos, action) for action in actionMoves]
        return possibleNextStates

    def __move(self, stateNum, action, statePos):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            Because all the exit statuses are treated as 1 single state, to enable the visualisation,
            requires to specify the state, which will be given as statePos
            ---
            Parameters
            ---

            :return next_stateNum, next_statePos
        """

        currentState = statePos
        # get the possible minotaur position
        minotaurPositions = self.__randomMove((currentState[2], currentState[3]))
        # randomly pick one minotaurPositions
        minotaurPos = random.sample(minotaurPositions, 1)[0]

        # if is Eaten or Exit remain in the stateNum, regardless of the action
        if (stateNum == self.STATE_EATEN or stateNum == self.STATE_EXIT):
            next_stateNum = stateNum
            agent_pos = (currentState[0], currentState[1])

        # general stateNum, choose one possible minotaur move and return corresponding stateNum
        else:
            agent_row = currentState[0] + self.actions[action][0];
            agent_col = currentState[1] + self.actions[action][1];

            # Is the future position an impossible one ?
            hitting_maze_walls = self.__isHittingWall((agent_row, agent_col))
            # Based on the impossiblity check return the next state.
            if hitting_maze_walls:
                agent_pos = (currentState[0], currentState[1])
            else:
                agent_pos = (agent_row, agent_col)
            next_stateNum = self.__getStateNum((*agent_pos, *minotaurPos))
        next_statePos = (*agent_pos, *minotaurPos)
        return next_stateNum, next_statePos

    def __moveResult(self, stateNum, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return a dictionary, includes the possible nextstates

            dictionary structure
            next_stateNum : [s1,s2,...]
            next_statePos : [(a_r, a_c, m_r1, m_c1), (a_r, a_c, m_r2, m_c2),...]
            action_rewards: [-1, -1,...]

        """
        # Compute the future position given current (stateNum, action)
        if (stateNum != self.STATE_EXIT and stateNum != self.STATE_EATEN):
            currentState = self.states[stateNum]
            agent_row = currentState[0] + self.actions[action][0];
            agent_col = currentState[1] + self.actions[action][1];

            hitting_maze_walls = self.__isHittingWall((agent_row, agent_col))
            # # Is the future position an impossible one ?
            # hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
            #                     (col == -1) or (col == self.maze.shape[1]) or \
            #                     (self.maze[row,col] == 1);

            # get the possible minotaur position
            minotaurPositions = self.__randomMove((currentState[2], currentState[3]))
            # Based on the impossiblity check return the next state.
            if hitting_maze_walls:
                agent_pos = (currentState[0], currentState[1])
                action_rewards = [self.IMPOSSIBLE_REWARD] * len(minotaurPositions)
            else:
                agent_pos = (agent_row, agent_col)
                action_rewards = [self.STEP_REWARD] * len(minotaurPositions)
            next_statePos = [(*agent_pos, *minotaur_pos) for minotaur_pos in minotaurPositions]
            # check if there are states of eaten

            #HERE
            next_stateNum = []
            index = 0
            for index in range(len(next_statePos)):
                # check if being eaten
                if (self.isEaten(next_statePos[index])):
                    #HERE
                    if next_stateNum == []:
                        next_statePos[index] = self.STATE_EATEN
                        #HERE
                        next_stateNum.append(self.STATE_EATEN)
                        action_rewards[index] = self.EATEN_REWARD
                # check if at exit
                elif (self.isExit(next_statePos[index])):
                    #HERE
                    if next_stateNum == []:
                        next_statePos[index] = self.STATE_EXIT
                        #HERE
                        next_stateNum.append(self.STATE_EXIT)
                        action_rewards[index] = self.STEP_REWARD
                    # action_rewards[index] = self.GOAL_REWARD
                else:
                    next_stateNum.append(self.map[next_statePos[index]])

        elif (stateNum == self.STATE_EXIT):
            next_statePos = [self.STATE_EXIT]
            next_stateNum = [self.STATE_EXIT]

            if (action != self.STAY):
                action_rewards = [self.STEP_REWARD]
            else:
                action_rewards = [self.GOAL_REWARD]
        else:
            # eaten stateNum
            next_statePos = [self.STATE_EATEN]
            next_stateNum = [self.STATE_EATEN]
            action_rewards = [self.EATEN_REWARD]

        moveResult = dict()
        moveResult['next_statePos'] = next_statePos
        moveResult['next_stateNum'] = next_stateNum
        moveResult['action_rewards'] = action_rewards
        return moveResult

    def __transitions(self):
        """ Computes the transition probabilities for every stateNum action pair.
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
                moveResult = self.__moveResult(s,a);
                next_stateNum = moveResult['next_stateNum']
                stateSize = len(next_stateNum)
                for next_s in next_stateNum:
                    transition_probabilities[next_s, s, a] = 1/stateSize;
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    moveResult = self.__moveResult(s,a);
                    action_rewards = moveResult['action_rewards']
                    # calculate the average of rewards
                    rewards[s,a] = sum(action_rewards) / len(action_rewards)

        # If the weights are described by a weight matrix
        else:
            raise Exception("Do not support weighted maze!")
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                     next_s = self.__move(s,a);
                     i,j = self.states[next_s];
                     # Simply put the reward as the weights o the next stateNum.
                     rewards[s,a] = weights[i][j];

        return rewards;

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current stateNum and time
            t = 0;
            s = self.map[start];
            # var that stores the current state pos
            currentStatePos = start
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next stateNum given the policy and the current stateNum
                # if multiple states, randomly chosen one
                next_s, next_statePos = self.__move(s,policy[s,t], currentStatePos);
                # Add the position in the maze corresponding to the next stateNum
                # to the path
                path.append(next_statePos)
                # Update time and stateNum for next iteration
                t +=1;
                s = next_s;
                currentStatePos = next_statePos
        if method == 'ValIter':
            # Initialize current stateNum, next stateNum and time
            t = 1;
            s = self.map[start];
            # var that stores the current state pos
            currentStatePos = start;
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next stateNum given the policy and the current stateNum
            next_s, next_statePos = self.__move(s,policy[s], currentStatePos);
            # Add the position in the maze corresponding to the next stateNum
            # to the path
            path.append(next_statePos);
            # Loop while stateNum is not the goal stateNum
            while s != next_s:
                # Update stateNum
                s = next_s;
                currentStatePos = next_statePos;
                # Move to next stateNum given the policy and the current stateNum
                next_s, next_statePos = self.__move(s,policy[s],currentStatePos);
                # Add the position in the maze corresponding to the next stateNum
                # to the path
                path.append(next_statePos)
                # Update time and stateNum for next iteration
                t +=1;
        return path


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
        :return numpy.array V     : Optimal values for every stateNum at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every stateNum,
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

    # # Map a color to each cell in the maze
    # col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED, 3: LIGHT_PURPLE};

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



def make_updateFunc(maze, path, minotaurMaze, grid, fig):
    """ for the animate function i is the time step
    """
    # Update the color at each frame
    def update(i):
        currentPos = path[i]
        # get agent position
        agentPos = (currentPos[0], currentPos[1])
        minotaurPos = (currentPos[2], currentPos[3])
        specialState = False
        if i > 0:

            if minotaurMaze.isExit(currentPos):
                grid.get_celld()[agentPos].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[agentPos].get_text().set_text('Player is out')
                specialState = True
            elif minotaurMaze.isEaten(currentPos):
                grid.get_celld()[minotaurPos].set_facecolor(LIGHT_RED)
                grid.get_celld()[minotaurPos].get_text().set_text('Player is eaten')
                specialState = True
            # reset the previous visited block
            # only reset if action is not stay, so previous and current do not coincide
            if (path[i-1][0] != agentPos[0] or path[i-1][1] != agentPos[1]):
                grid.get_celld()[(path[i-1][0], path[i-1][1])].set_facecolor(col_map[maze[path[i-1][0],path[i-1][1]]])
                grid.get_celld()[(path[i-1][0], path[i-1][1])].get_text().set_text('')
            grid.get_celld()[(path[i-1][2], path[i-1][3])].set_facecolor(col_map[maze[path[i-1][2], path[i-1][3]]])
            grid.get_celld()[(path[i-1][2], path[i-1][3])].get_text().set_text('')
        # it is possible that minotaur occupied previously the current agent position
        # so first eliminate the history minotaur position, then plot the agent
        # plot agent position if not special state
        if not specialState :
            grid.get_celld()[agentPos].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[agentPos].get_text().set_text('Player')
        # plot minotaur position
        grid.get_celld()[minotaurPos].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[minotaurPos].get_text().set_text('Mino')
        return fig, grid

def animate_solution(maze, path, minotaurMaze, createGIF = True):

    # # Map a color to each cell in the maze
    # col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

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



    if createGIF :
        update = make_updateFunc(maze, path, minotaurMaze, grid, fig)
        anim = FuncAnimation(fig, update, frames=np.arange(0, len(path)), interval=200)
        anim.save('maze.gif', dpi=80, writer='imagemagick')
    else :

        # Update the color at each frame
        for i in range(len(path)):
            currentPos = path[i]
            # get agent position
            agentPos = (currentPos[0], currentPos[1])
            minotaurPos = (currentPos[2], currentPos[3])
            specialState = False
            if i > 0:

                if minotaurMaze.isExit(currentPos):
                    grid.get_celld()[agentPos].set_facecolor(LIGHT_GREEN)
                    grid.get_celld()[agentPos].get_text().set_text('Player is out')
                    specialState = True
                elif minotaurMaze.isEaten(currentPos):
                    grid.get_celld()[minotaurPos].set_facecolor(LIGHT_RED)
                    grid.get_celld()[minotaurPos].get_text().set_text('Player is eaten')
                    specialState = True
                # reset the previous visited block
                # only reset if action is not stay, so previous and current do not coincide
                if (path[i-1][0] != agentPos[0] or path[i-1][1] != agentPos[1]):
                    grid.get_celld()[(path[i-1][0], path[i-1][1])].set_facecolor(col_map[maze[path[i-1][0],path[i-1][1]]])
                    grid.get_celld()[(path[i-1][0], path[i-1][1])].get_text().set_text('')
                grid.get_celld()[(path[i-1][2], path[i-1][3])].set_facecolor(col_map[maze[path[i-1][2], path[i-1][3]]])
                grid.get_celld()[(path[i-1][2], path[i-1][3])].get_text().set_text('')
            # it is possible that minotaur occupied previously the current agent position
            # so first eliminate the history minotaur position, then plot the agent
            # plot agent position if not special state
            if not specialState :
                grid.get_celld()[agentPos].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[agentPos].get_text().set_text('Player')
            # plot minotaur position
            grid.get_celld()[minotaurPos].set_facecolor(LIGHT_PURPLE)
            grid.get_celld()[minotaurPos].get_text().set_text('Mino')

            #display.display(fig)
            #display.clear_output(wait=True)
            #time.sleep(1)
# test
            plt.pause(1)
        plt.show()

if __name__ == "__main__" :

    # change to the file position
    #os.chdir(os.path.abspath(__file__))

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
