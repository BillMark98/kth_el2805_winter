# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518      
import numpy as np
import matplotlib.pyplot as plt
import time
import operator
import random
from IPython import display
from matplotlib.animation import FuncAnimation

import os

def change2FileDir():
    """ change working directory to the current file directory"""
    # change to the file position
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Implemented methods
methods = ['DynProg', 'ValIter', 'QLearn', 'SARSA'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';
LIGHT_BLUE   = '#ADD8E6';
BLUE_VIOLET  = '#8A2BE2';

col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED, 3: LIGHT_PURPLE, 4: LIGHT_BLUE};

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

    # maze value
    EXIT_POSITION_MAZE_VALUE = 2
    KEY_POSITION_MAZE_VALUE = 4

    # for the random move of the minotaur with key picking
    PROB_TOWARD_AGENT = 0.35

    def __init__(self, maze, weights=None, random_rewards=False, keyPicking=False, greedyGoal = False):
        """ Constructor of the environment Maze.

        maze convention:
            0 : free square to move
            1 : wall
            2 : exit
            3 : minotaur
            4 : location of key
        keyPicking : indicate whether first need to pick up the key

        greedyGoal : the boolean indicate whether the agent wants to achieve the goal as quickly as possible
         default false, this will impact the calculation of reward function
        if set to False(default), the moment one hits exit, will receive a reward, and keep that reward (which is in fact no loss, so 0) 
        if set to True, the moment one hits exit, one still gets a step reward cost, and will only keep the reward if one keeps in that state,
        so this will force the agent to move to exit as long it is near the exit.

        """
        self.maze                     = maze;
        self.keyPicking               = keyPicking;
        self.greedyGoal               = greedyGoal
                                                          
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
        # first check if eaten
        if not self.isEaten(pos_state):
            if (self.keyPicking):
                return (self.maze[pos_state[0], pos_state[1]] == self.EXIT_POSITION_MAZE_VALUE) and pos_state[4]
            else:
                return self.maze[pos_state[0], pos_state[1]] == self.EXIT_POSITION_MAZE_VALUE
        else:
            return False
    
    def isKeyPickingPos(self, pos_state):
        """ given position state, test if it is the key position
            
        ---
        Exception
        ---
            raise Exception if keyPicking is false

        """

        if (not self.keyPicking):
            raise Exception("no key to pick, isKeyPickingPos invalid")

        # check if Eaten
        if not self.isEaten(pos_state):
            return self.maze[pos_state[0], pos_state[1]] == self.KEY_POSITION_MAZE_VALUE
        else:
            return False

    def __setCoordValue(self, pos_state, index = 4, value = 1):
        """ help function to set one position of a given position state to a given value
            e.g. used for keyPicking
        """
        # first convert to a list
        pos_state = list(pos_state)
        pos_state[index] = value
        # then convert back
        pos_state = tuple(pos_state)
        return pos_state

    def isTerminalStateNum(self, stateNum):
        """ given state number, return if it is terminal state"""
        return (stateNum == self.STATE_EXIT) or (stateNum == self.STATE_EATEN)

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
        # if no key picking

        if (not self.keyPicking):
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
        else:
            # stateNum count from 2, the first two reserved
            s = 2;
            states[0] = 0;
            states[1] = 1;
            # at most S * S * 2 states,
            # S is the number of squares of the maze
            # pos_state is of the form (i,j,mx,my, keyPicked), where (i,j) is the position of the explorer theseus (mx,my) pos of minotaur,
            # and keyPicked indicate whether the key is keyPicked
            for keyPicked in range(2):            
                for i in range(self.maze.shape[0]):
                    for j in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            for mx in range(self.maze.shape[0]):
                                for my in range(self.maze.shape[1]):
                                    if not (self.isEaten((i,j,mx,my, keyPicked)) or self.isExit((i,j,mx,my, keyPicked))):
                                        # account state if 
                                        # 1) a normal state (not exit, not eaten, not keypicking)
                                        # 2) keyPicking but with keyPicked == 1  (since it will directly transform to a keypicked state, so these two states degenerate into one)
                                        if(not (self.isKeyPickingPos((i,j,mx,my,keyPicked)) and keyPicked == 0)):
                                            states[s] = (i,j,mx,my,keyPicked);
                                            map[(i,j, mx,my, keyPicked)] = s;
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
            if (self.keyPicking and self.isKeyPickingPos(pos_state)):
                pos_state = self.__setCoordValue(pos_state)
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

    def __randomMove(self, minotaurPos, agentPos = ""):
        """ randomMove, use to simulate the move of minotaur.
            if no constraint, choose one of the four randomly, otherwise, choose the possible moves uniform randomly

            if keyPicking is True, then will use agent position to further specify minotaur move, see problem description

            --- 
            Parameters
            ---
            minotaurPos : a 2-element tuple, the usual coordinate of a point
            agentPos : optional, only used when keyPicking is true
            ---
            return
            ---
            return a list of possible states
        """
        row = minotaurPos[0]
        col = minotaurPos[1]
        possibleActions = set([self.MOVE_LEFT,self.MOVE_RIGHT, self.MOVE_UP, self.MOVE_DOWN])

        if row == 0:
            possibleActions.discard(self.MOVE_UP)
        elif row == self.maze.shape[0] - 1 :
            possibleActions.discard(self.MOVE_DOWN)

        if col == 0:
            possibleActions.discard(self.MOVE_LEFT)
        elif col == self.maze.shape[1] - 1 :
            possibleActions.discard(self.MOVE_RIGHT)

        if (self.keyPicking):
            # decide whether move toward agent
            random_number = np.random.rand()
            if random_number < self.PROB_TOWARD_AGENT :
                # agent not at the right of the minotaur, so delete move right
                if (agentPos[1] <= col) :
                    possibleActions.discard(self.MOVE_RIGHT)

                # agent not at left, so delete MOVE_LEFT
                # use a different if condition in case agentPos at same col as minotaur
                if (agentPos[1] >= col):
                    possibleActions.discard(self.MOVE_LEFT)
                
                # agent not below of minotaur, so delete move down
                if (agentPos[0] <= row):
                    possibleActions.discard(self.MOVE_DOWN)
                
                # agent not above, so delete MOVE_UP
                if (agentPos[0] >= row):
                    possibleActions.discard(self.MOVE_UP)
        if (len(possibleActions) == 0):
            print("__randomMove error!")
            print("minotaur position ({0},{1})".format(minotaurPos[0], minotaurPos[1]))
            raise Exception("No actions for minotaur possible, something is wrong!")

        # get the actionMoves
        actionMoves = [self.actions[actionName] for actionName in possibleActions]

        possibleNextStates = [self.__coordinateAddition(minotaurPos, action) for action in actionMoves]
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
        if (self.keyPicking):
            minotaurPositions = self.__randomMove((currentState[2], currentState[3]), (currentState[0], currentState[1]))
            keyPicked = currentState[4]
        else:
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
            if (self.keyPicking):
                # update the 5-th coordinate
                combined_statePos = (*agent_pos, *minotaurPos, currentState[4])
                if (self.isKeyPickingPos(combined_statePos)):
                    keyPicked = 1
                    combined_statePos = self.__setCoordValue(combined_statePos)
            else:
                combined_statePos = (*agent_pos, *minotaurPos)
            next_stateNum = self.__getStateNum(combined_statePos)
        if (self.keyPicking):
            next_statePos = (*agent_pos, *minotaurPos, keyPicked)
        else:
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
            state_transitionProb : { 0 : 1/4, ...}

            Note that it is possible there are multiple same states, so the probability is not uniform,
            but need to sum all the occurences of one state and divided by the whole length. This does not apply to the reward calculation,
            because the multiplicity will be considered during the sum up, see the __rewards function
        """

        # state transition probability dictionary
        state_transitionProb = dict()

        # Compute the future position given current (stateNum, action)
        # if a normal state
        if (stateNum != self.STATE_EXIT and stateNum != self.STATE_EATEN):
            currentState = self.states[stateNum]

            if (self.keyPicking):
                keyPicked = currentState[4]               
            agent_row = currentState[0] + self.actions[action][0];
            agent_col = currentState[1] + self.actions[action][1];

            hitting_maze_walls = self.__isHittingWall((agent_row, agent_col))
            # # Is the future position an impossible one ?
            # hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
            #                     (col == -1) or (col == self.maze.shape[1]) or \
            #                     (self.maze[row,col] == 1);

            # get the possible minotaur position
            if (self.keyPicking):
                minotaurPositions = self.__randomMove((currentState[2], currentState[3]), (currentState[0], currentState[1]))
            else:
                minotaurPositions = self.__randomMove((currentState[2], currentState[3]))            
            # Based on the impossiblity check return the next state.
            if hitting_maze_walls:
                agent_pos = (currentState[0], currentState[1])
                action_rewards = [self.IMPOSSIBLE_REWARD] * len(minotaurPositions)
            else:
                agent_pos = (agent_row, agent_col)
                action_rewards = [self.STEP_REWARD] * len(minotaurPositions)
            if (self.keyPicking):
                next_statePos = [None] * len(minotaurPositions)
                # need to check for each single state whether it is key Picked or not
                for minotaur_pos, index in zip(minotaurPositions, list(range(len(minotaurPositions)))):
                    tempStatePos = (*agent_pos, *minotaur_pos, keyPicked)
                    # temperory keyPicked flag, if a new picking state hit, then update to 1
                    # if not, then just inherit from the previous state
                    tempKeyPicked = keyPicked
                    if (self.isKeyPickingPos(tempStatePos)):
                        tempKeyPicked = 1
                    next_statePos[index] = (*agent_pos, *minotaur_pos, tempKeyPicked)
            else:
                next_statePos = [(*agent_pos, *minotaur_pos) for minotaur_pos in minotaurPositions]
            # check if there are states of eaten
            next_stateNum = [None] * len(next_statePos)
            index = 0
            for index in range(len(next_statePos)):
                
                # check if being eaten
                if (self.isEaten(next_statePos[index])):
                    next_statePos[index] = self.STATE_EATEN
                    next_stateNum[index] = self.STATE_EATEN
                    action_rewards[index] = self.EATEN_REWARD
                    if (self.STATE_EATEN not in state_transitionProb):
                        state_transitionProb[self.STATE_EATEN] = 1
                    else:
                        state_transitionProb[self.STATE_EATEN] += 1
                # check if at exit
                elif (self.isExit(next_statePos[index])):
                    next_statePos[index] = self.STATE_EXIT
                    next_stateNum[index] = self.STATE_EXIT
                    if (self.greedyGoal):
                        action_rewards[index] = self.STEP_REWARD
                    else:
                        action_rewards[index] = self.GOAL_REWARD
                    if (self.STATE_EXIT not in state_transitionProb):
                        state_transitionProb[self.STATE_EXIT] = 1
                    else:
                        state_transitionProb[self.STATE_EXIT] += 1
                else:
                    next_stateNum[index] = self.map[next_statePos[index]]
                    if (next_stateNum[index] in state_transitionProb):
                        # an error occurred, because this state should be unique
                        raise Exception("normal state visted more than once!")
                    state_transitionProb[next_stateNum[index]] = 1
            # update the prob
            for key in state_transitionProb.keys():
                # calculate the probability
                state_transitionProb[key] /= len(next_statePos)
            
        elif (stateNum == self.STATE_EXIT):
            next_statePos = [self.STATE_EXIT]
            next_stateNum = [self.STATE_EXIT]

            if (self.greedyGoal):
                if (action != self.STAY):
                    action_rewards = [self.STEP_REWARD]
                else:
                    action_rewards = [self.GOAL_REWARD]
            else:
                action_rewards = [self.STEP_REWARD]
            state_transitionProb[self.STATE_EXIT] = 1
        else:
            # eaten stateNum
            next_statePos = [self.STATE_EATEN]
            next_stateNum = [self.STATE_EATEN]
            action_rewards = [self.EATEN_REWARD]
            state_transitionProb[self.STATE_EXIT] = 1

        
        moveResult = dict()
        moveResult['next_statePos'] = next_statePos
        moveResult['next_stateNum'] = next_stateNum
        moveResult['action_rewards'] = action_rewards
        moveResult['state_transitionProb'] = state_transitionProb
        return moveResult

    def __transitions(self):
        """ Computes the transition probabilities for every stateNum action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. 
        for s in range(self.n_states):
            for a in range(self.n_actions):
                moveResult = self.__moveResult(s,a);
                # next_stateNum = moveResult['next_stateNum']
                state_transitionProb = moveResult['state_transitionProb']
                for next_s in state_transitionProb.keys():
                    transition_probabilities[next_s, s, a] = state_transitionProb[next_s];
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

    def getNextStateNum(self, currentStateNum, currentState_actionNum):
        """ given current state num and action num, return the next state num
            Note there are several possible next state due to the random move of the minotaur
            so will randomly choose one

        ---
        Return
        ----
            return state number of next possible state
        
        ---
        Exception
        ---
            throw exceptions if currentStateNum is terminal state
        """
        if (self.isTerminalStateNum(currentStateNum)):
            raise Exception("state number is terminal state, impossible to derive next move due to degeneracy of states")

        nextStateNum, _ = self.__move(currentStateNum, currentState_actionNum, self.states[currentStateNum])
        return nextStateNum

    def simulate(self, start, policy, method, **kargs):
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
            while t <= horizon-1:
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

            try:
                # get the probability to survive
                probability_to_survive = kargs["prob"]
            except KeyError:
                print("To simulate value iteration, need to specify prob to survive")
                return []
            else:
                
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
                # Loop while stateNum is not terminal state or the time is before ttl
                while not self.isTerminalStateNum(next_s):
                    # Update stateNum
                    s = next_s;
                    currentStatePos = next_statePos;
                    random_number = np.random.rand()
                    if random_number>probability_to_survive:
                        break
                    else:

                        # Move to next stateNum given the policy and the current stateNum
                        next_s, next_statePos = self.__move(s,policy[s],currentStatePos);
                        # Add the position in the maze corresponding to the next stateNum
                        # to the path
                        path.append(next_statePos)
                        # Update time and stateNum for next iteration
                        t +=1;
            return path
        if method == "QLearn" or method == "SARSA":
            try:
                # get the probability to survive
                probability_to_survive = kargs["prob"]
            except KeyError:
                print("To simulate q learning, need to specify prob to survive")
                return []
            else:
                
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
                # Loop while stateNum is not terminal state or the time is before ttl
                while not self.isTerminalStateNum(next_s):
                    # Update stateNum
                    s = next_s;
                    currentStatePos = next_statePos;
                    random_number = np.random.rand()
                    if random_number>probability_to_survive:
                        break
                    else:

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


def qLearning(env, gamma, epsilon, learningRateFunc, episodes = 50000):
    """ implements the q-learning algorithms
    
    ---
    Parameters:
    ---

        learningRateFunc: a function, given n (number of visited) will output the learning rate
        where n is of the form n(s,a)  (function of BOTH state AND action)

        episodes: the number of episodes to complete
    ---
    Return
    ----

        Value function V (n_states * 1) policy (n_sates * 1), iterationCounter(episodes * 1), which count how many iterations take for each
        episode
    """
    # The q-learning algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    # - learning rate function
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the QL to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    # Iteration counter for each episode
    iterationCounter   = [0] * episodes;

    # Value Function over episodes, to see the convergence speed
    V_over_episodes = np.zeros((n_states, episodes))
    # help function to get next move
    def nextMoveEpsSoft(stateNum, eps = epsilon / n_actions):
        """ return next action number with epsilon soft policy"""
        # choose random action if random gives a value less than eps
        if np.random.random() < eps:
            return random.sample(range(n_actions), 1)[0]
        else :
            # choose the greedy
            return np.argmax(Q[stateNum,:])
    # get starting position for a new episode:
    def getStartingStateNum():
        return random.randint(0, n_states - 1)
        
    # Initialization of the Q function
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    
    # begin episodes
    for episode in range(episodes):
        # clear all the counts
        stateVisits = np.zeros((n_states,n_actions))
        # get starting state number
        futureStateNum = getStartingStateNum()
        current_loopCount = 0
        while(not env.isTerminalStateNum(futureStateNum)):
            currentStateNum = futureStateNum
            # get next move
            currentState_actionNum = nextMoveEpsSoft(currentStateNum)
            # add visit count
            stateVisits[currentStateNum, currentState_actionNum] += 1
                        
            # get next state number
            futureStateNum = env.getNextStateNum(currentStateNum, currentState_actionNum)
            # get the reward of the currentState_actionNum
            move_reward = env.rewards[currentStateNum, currentState_actionNum]
            # get the old q value
            old_qValue = Q[currentStateNum, currentState_actionNum]
            # calculate the difference
            temporal_difference = move_reward + gamma * np.max(Q[futureStateNum,:]) - old_qValue
            # update the q value
            Q[currentStateNum, currentState_actionNum] = old_qValue + learningRateFunc(stateVisits[currentStateNum, currentState_actionNum]) * temporal_difference
            current_loopCount += 1
        iterationCounter[episode] = current_loopCount
        V_over_episodes[:, episode] = np.max(Q,1)
    # compute the Value Function
    V = np.max(Q, 1)
    # compute the policy
    policy = np.argmax(Q, 1)
    return V, policy, iterationCounter, V_over_episodes

def sarsa(env, gamma, epsilon, learningRateFunc, episodes = 50000):
    """ implements the sarsa algorithms
    
    ---
    Parameters:
    ---

        learningRateFunc: a function, given n (number of visited) will output the learning rate
        where n is of the form n(s,a)  (function of BOTH state AND action)

        episodes: the number of episodes to complete
    ---
    Return
    ----

        Value function V (n_states * 1) policy (n_sates * 1), iterationCounter(episodes * 1), which count how many iterations take for each
        episode
    """
    # The sarsa algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    # - learning rate function
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the QL to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    # Iteration counter for each episode
    iterationCounter   = [0] * episodes;

    # Value Function over episodes, to see the convergence speed
    V_over_episodes = np.zeros((n_states, episodes))
    # help function to get next move
    def nextMoveEpsSoft(stateNum, eps = epsilon / n_actions):
        """ return next currentState_actionNum with epsilon soft policy"""
        # choose random action if random gives a value less than eps
        if np.random.random() < eps:
            return random.sample(range(n_actions), 1)[0]
        else :
            # choose the greedy
            return np.argmax(Q[stateNum,:])
    # get starting position for a new episode:
    def getStartingStateNum():
        return random.randint(0, n_states - 1)
        
    # Initialization of the Q function
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    
    # begin episodes
    for episode in range(episodes):
        # clear all the counts
        stateVisits = np.zeros((n_states,n_actions))
        # get starting state number
        futureStateNum = getStartingStateNum()
        current_loopCount = 0
        while(not env.isTerminalStateNum(futureStateNum)):
            currentStateNum = futureStateNum
            # get next move of the old state
            currentState_actionNum = nextMoveEpsSoft(currentStateNum)
            # add visit count
            stateVisits[currentStateNum, currentState_actionNum] += 1
                        
            # get next state number
            futureStateNum = env.getNextStateNum(currentStateNum, currentState_actionNum)
            # get next action
            futureState_actionNum = nextMoveEpsSoft(futureStateNum)
            # get the reward of the currentState_actionNum
            move_reward = env.rewards[currentStateNum, currentState_actionNum]
            # get the old q value
            old_qValue = Q[currentStateNum, currentState_actionNum]
            # calculate the difference
            temporal_difference = move_reward + gamma * Q[futureStateNum, futureState_actionNum] - old_qValue
            # update the q value
            Q[currentStateNum, currentState_actionNum] = old_qValue + learningRateFunc(stateVisits[currentStateNum, currentState_actionNum]) * temporal_difference
            current_loopCount += 1
        iterationCounter[episode] = current_loopCount
        V_over_episodes[:, episode] = np.max(Q,1)
    # compute the Value Function
    V = np.max(Q, 1)
    # compute the policy
    policy = np.argmax(Q, 1)
    return V, policy, iterationCounter, V_over_episodes



def exit_probability(env, startPos, method = "DynProg", **kargs):
    """ given the maze, the startPos, time horizon, calculate the exit probability
    """


    start_vector = np.zeros((env.n_states, 1))
    start_vector[startPos] = 1.
    current_state = start_vector
    if method == "DynProg":
        try:
            time_horizon = kargs["time_horizon"]
        except KeyError:
            print("Need to speicify time horizon for dynmaic programming")
            pass
        else:
            V, policy= dynamic_programming(env, time_horizon)

            print("For T = ", str(time_horizon) + ": ")
            for t in range(1, time_horizon+1):
                if time_horizon==16 and t == 15:
                    for i in range(len(current_state)):
                        print(i, ": ", current_state[i])
                    print(np.sum(current_state))
                    new_state = np.zeros((env.n_states, 1))
                    #go through all possible states (basically we loop through current_state)
                    for state in range(env.n_states):
                        #if the probability of being in this state is 0, dont bother
                        #to calculate (saves some time, we dont need this condition)
                        if current_state[state]!=0.:
                                #the agent will perform the action according to the policy
                            action_done = policy[state][t]
                            print(action_done)
                                #create one hot encoding of this state to add to new_state
                            state_vector = np.zeros((env.n_states, 1))
                            state_vector[state] = 1.

                                #add to new state. current_state[state] is the probability of being
                                #in state state
                            print(np.matmul(env.transition_probabilities[:, :, int(action_done)], state_vector))
                            new_state += current_state[state]*np.matmul(env.transition_probabilities[:, :, int(action_done)], state_vector)


                else:
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
                    
    elif method == "ValIter":
        try:
            gamma = kargs["gamma"]
            epsilon = kargs["epsilon"]
        except KeyError:
            print("Need to specify the gamma and epsilon for value Iteration")
            pass
        else:
            V, policy = value_iteration(env, gamma, epsilon)
            # todo
            # analytical solution possible????
    

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

def animate_solution(maze, path, minotaurMaze, createGIF = True, saveFigName = "maze.gif"):

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

    def update(i):
        currentPos = path[i]
        # get agent position
        agentPos = (currentPos[0], currentPos[1])
        minotaurPos = (currentPos[2], currentPos[3])
        specialState = False
        if (minotaurMaze.keyPicking):
            keyPicked = currentPos[4]
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
            if (minotaurMaze.keyPicking and (minotaurMaze.isKeyPickingPos(currentPos) or keyPicked)):
                grid.get_celld()[agentPos].set_facecolor(BLUE_VIOLET)
            else:
                grid.get_celld()[agentPos].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[agentPos].get_text().set_text('Player')
        # plot minotaur position
        grid.get_celld()[minotaurPos].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[minotaurPos].get_text().set_text('Mino')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)          
    
    if createGIF :
        anim = FuncAnimation(fig, update, frames=np.arange(0, len(path)), interval=500)
        anim.save(saveFigName, dpi=80, writer='imagemagick')
    else :    
        for i in range(len(path)):
            update(i)
            # display.display(fig)
            # display.clear_output(wait=True)
            # time.sleep(1)        
# test

if __name__ == "__main__" :

    # change to the file position
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

    # # Finite horizon
    # horizon = 20
    # # Solve the MDP problem with dynamic programming 
    # V, policy= dynamic_programming(env,horizon)
    # # Simulate the shortest path starting from position A
    # method = 'DynProg';
    # start  = (0,0,6,5);
    # path = env.simulate(start, policy, method);    
    # animate_solution(maze, path, env)

    # # simulate value iteration
    # life_expectancy = 30
    # gamma = 1 - 1/life_expectancy
    # epsilon = 0.1

    # V, policy = value_iteration(env, gamma, epsilon)
    # method = 'ValIter';
    # start  = (0,0,6,5);
    # path = env.simulate(start, policy, method, prob = gamma);    
    # # print(path)
    # animate_solution(maze, path, env,saveFigName = "mazeValIter.gif")


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

    env = Maze(maze,keyPicking=True)    

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
    animate_solution(maze, path, env,saveFigName = "mazeValIter_keyPicking.gif")

    # output the V and policy
    np.savetxt("valueIter_V.txt", V, fmt = "%5.4f")
    np.savetxt("valIter_policy.txt", policy, fmt = "%5d")

    # episodes = 500

    # V, policy, iteration_counter, V_over_episodes = qLearning(env, gamma, epsilon, lambda x : 1 / np.float_power(x, alpha), episodes)
    # method = 'QLearn';
    # start  = (0,0,6,5,0);
    # path = env.simulate(start, policy, method, prob = gamma);    
    # # print(path)
    # animate_solution(maze, path, env,saveFigName = "mazeQLearn_temp.gif")    
    # np.savetxt("qLearn_V_temp.txt", V, fmt = "%5.4f")
    # np.savetxt("qLearn_policy_temp.txt", policy, fmt = "%5d")

    # # plot the value function of the first state over time
    # startNum = env.map[start]
    # plt.plot(list(range(episodes)), V_over_episodes[startNum,:])
    # plt.xlabel("episodes")
    # plt.ylabel("value function")
    # plt.title("value function over episode")
    # plt.savefig("v_over_eps_qLearn_temp.png")
    # plt.show()