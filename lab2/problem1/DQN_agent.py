# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
from torch._C import Value
from torch.autograd import grad
# torch neuron network
import torch.nn as nn
from torch.nn.modules import loss
from ReplayBuffer import ExperienceReplayBuffer, Experience
import torch.optim as optim
import random

class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass


    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action

class DQN_NeuronNetwork(nn.Module):
    ''' neuron network for dqn'''
    
    def __init__(self, n_actions: int, n_inputs = 8, layers = 2, neuronNums= [6,6], dueling = True):
        '''

        ----
        Parameters:
        ----

        n_inputs : number of inputs, default 8  (state space dimension)

        layers:  number of layers

        neurons: list each element is the number of neurons per layer

        dueling: boolean to indicate whether to use dueling dqn
        '''
        super().__init__()
        self.n_actions = n_actions
        self.last_action = None
        self.layers = layers
        self.neuronNums = neuronNums
        # self.linearLists = [None] * (self.layers + 1)
        # self.actLists = [None] * self.layers
        inputCount = n_inputs
        n_outputs = n_actions
        self.dueling = dueling

        # for index in range(layers):
        #     self.linearLists[index] = nn.Linear(inputCount, neuronNums[index])
        #     self.actLists[index] = nn.ReLU()
        #     inputCount = neuronNums[index]
        #     if (index == layers - 1):
        #         self.linearLists[index + 1] = nn.Linear(inputCount, n_outputs)
        self.linear1 = nn.Linear(n_inputs, neuronNums[0])
        self.act1 = nn.ReLU()
        if (layers == 2):
            self.linear2 = nn.Linear(neuronNums[0], neuronNums[1])
            self.act2 = nn.ReLU()
            if (dueling):
                self.linear3 = nn.Linear(neuronNums[1], n_actions + 1)   # add an extra V(s)
                # self.act3 = nn.ReLU()
            else:
                self.linear3 = nn.Linear(neuronNums[1], n_actions)
        else:
            if (dueling):
                raise Exception("Not implemented dueling for single layer yet")
            else:
                self.linear2 = nn.Linear(neuronNums[0], n_actions)
            

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        meta_state = state
        # for index in range(self.layers):
        #     meta_state = self.linearLists[index](meta_state)
        #     meta_state = self.actLists[index](meta_state)
        meta_state = self.linear1(meta_state)
        meta_state = self.act1(meta_state)
        meta_state = self.linear2(meta_state)
        if (self.layers == 2):
            meta_state = self.act2(meta_state)
            meta_state = self.linear3(meta_state)
            if (self.dueling):
                # create tensor for column operation
                # get_Q_func_matrix = torch.ones((2,), dtype = torch.float32, requires_grad=False)
                get_Q_func_matrix = torch.full((self.n_actions, self.n_actions), -1/self.n_actions)
                identity_matrix = torch.eye(self.n_actions, dtype = torch.float32)
                get_Q_func_matrix = get_Q_func_matrix + identity_matrix
                V_array = torch.ones((1,self.n_actions), dtype = torch.float32)
                get_Q_func_matrix = torch.cat((get_Q_func_matrix, V_array), dim = 0)
                meta_state = torch.matmul(meta_state, get_Q_func_matrix)

        state = meta_state
        return state


class DQN_agent(Agent):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int, discount_factor = 0.95, buffer_size = 10000,  train_batch_size = 128,
    episodes = 1000, target_freq_update = 50, learning_rate = 1e-3, n_inputs = 8, layers = 2, neuronNums= [6,6]):
        '''

        ----
        Parameters:
        ----

        n_inputs : number of inputs, default 8  (state space dimension)

        layers:  number of layers

        neurons: list each element is the number of neurons per layer
        '''
        super.__init__()
        self.n_actions = n_actions
        self.last_action = None
        self.layers = layers
        self.neuronNums = neuronNums
        self.learning_rate = learning_rate
        # self.linearLists = [None] * (self.layers + 1)
        # self.actLists = [None] * self.layers
        # inputCount = n_inputs
        # n_outputs = n_actions
        
        self.neuronNetworks = DQN_NeuronNetwork(n_actions, n_inputs, layers, neuronNums=neuronNums)
        self.optimizer = optim(self.neuronNetworks.parameters(), lr = learning_rate)
        # for index in range(layers):
        #     self.linearLists[index] = nn.Linear(inputCount, neuronNums[index])
        #     self.actLists[index] = nn.ReLU()
        #     inputCount = neuronNums[index]
        #     if (index == layers - 1):
        #         self.linearLists[index] = nn.Linear(inputCount, n_outputs)

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''

        return state

    def backward(self):
        ''' Performs a backward pass on the network '''
        # target Value
        # ..
        # loss
        # optimitiyer.yero grad
        # loss.backward
        # optimiyer.step