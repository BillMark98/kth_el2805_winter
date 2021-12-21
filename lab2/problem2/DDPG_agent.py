# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518
#  

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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 26th October 2020, by alessior@kth.se
#



# same from problem 1

# Load packages
from pickle import FALSE
import numpy as np
from numpy.core import multiarray
import torch
from torch._C import Value
from torch.autograd import grad
# torch neuron network
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import random

from DDPG_soft_updates import soft_updates

import gym
from collections import deque, namedtuple
import copy
# copied from lab0:

### Experience class ###

# namedtuple is used to create a special type of tuple object. Namedtuples
# always have a specific name (like a class) and specific fields.
# In this case I will create a namedtuple 'Experience',
# with fields: state, action, reward,  next_state, done.
# Usage: for some given variables s, a, r, s, d you can write for example
# exp = Experience(s, a, r, s, d). Then you can access the reward
# field by  typing exp.reward
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

N_episodes = 1000                             # Number of episodes
eps_max = 0.99
eps_min = 0.05
Z = np.floor(0.9 * N_episodes)


# # method to decay epsilon
# eps_decay_method = 2

# def epsilonDecay(method = eps_decay_method):
#     ''' the function to update the epsilon, function of episode '''
#     def eps_decay(episode):
#         if (method == 1):
#             episode = np.max([eps_min, eps_max - ((eps_max - eps_min) * (episode - 1) / (Z - 1))])
#         else:
#             episode = np.max([eps_min, eps_max * np.float_power(eps_min / eps_max, (episode - 1) / (Z - 1))])
#         return episode

#     return eps_decay

# PRINT_GRAD = True
PRINT_GRAD = False



class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length, CER = True):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)
        self.last_entry = None
        self.last_entry_index = None
        self.cer = CER # whether to use CER or not

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)
        self.last_entry = experience


    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )
        if self.cer:
            indices = set(indices)
            # the experience has the last len index
            indices.add(len(self.buffer) - 1)


        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)
    # def write2File(fileName = "stored_buffer.txt"):
    #     with open(fileName, "w") as f:
    #         f.write()




class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

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

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)


class DDPG_NeuronNetwork_Actor(nn.Module):
    ''' neuron network for ddpg actor'''
    
    def __init__(self, n_actions: int, n_inputs = 8, layers = 3, neuronNums= [400,200,200]):
    # def __init__(self, n_actions: int, n_inputs = 8, layers = 3, neuronNums= [400,200,200], dueling = True):
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

        # # experience buffer variables
        # self.dueling = dueling

        # for index in range(layers):
        #     self.linearLists[index] = nn.Linear(inputCount, neuronNums[index])
        #     self.actLists[index] = nn.ReLU()
        #     inputCount = neuronNums[index]
        #     if (index == layers - 1):
        #         self.linearLists[index + 1] = nn.Linear(inputCount, n_outputs)
        self.linear1 = nn.Linear(n_inputs, neuronNums[0])
        self.act1 = nn.ReLU()
        # if (layers != 3):
        #     raise Exception("layers other than 3 not supported")
        self.linear2 = nn.Linear(neuronNums[0], neuronNums[1])
        self.act2 = nn.ReLU()
        if (layers == 3):
            self.linear3 = nn.Linear(neuronNums[1], neuronNums[2])
            self.act3 = nn.ReLU()
            self.linear4 = nn.Linear(neuronNums[2],n_outputs)
            # extra tanh
            self.act4 = nn.Tanh()
        elif (layers == 2):
            self.linear3 = nn.Linear(neuronNums[1], n_outputs)
            self.act3 = nn.Tanh()

        # # for cuda

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.to(self.device)

        # if (layers == 2):
        #     self.linear2 = nn.Linear(neuronNums[0], neuronNums[1])
        #     self.act2 = nn.ReLU()
        #     if (dueling):
        #         self.linear3 = nn.Linear(neuronNums[1], n_actions + 1)   # add an extra V(s)
        #         # self.act3 = nn.ReLU()
        #     else:
        #         self.linear3 = nn.Linear(neuronNums[1], n_actions)
        # else:
        #     if (dueling):
        #         raise Exception("Not implemented dueling for single layer yet")
        #     else:
        #         self.linear2 = nn.Linear(neuronNums[0], n_actions)
            

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        meta_state = state
        # for index in range(self.layers):
        #     meta_state = self.linearLists[index](meta_state)
        #     meta_state = self.actLists[index](meta_state)
        meta_state = self.linear1(meta_state)
        meta_state = self.act1(meta_state)
        meta_state = self.act2(self.linear2(meta_state))
        meta_state = self.act3(self.linear3(meta_state))
        if (self.layers == 3):
            meta_state = self.act4(self.linear4(meta_state))
        state = meta_state
        return state
        
        # if (self.layers == 2):
        #     meta_state = self.act2(meta_state)
        #     meta_state = self.linear3(meta_state)
        #     if (self.dueling):
        #         # create tensor for column operation
        #         # get_Q_func_matrix = torch.ones((2,), dtype = torch.float32, requires_grad=False)
        #         get_Q_func_matrix = torch.full((self.n_actions, self.n_actions), -1/self.n_actions)
        #         identity_matrix = torch.eye(self.n_actions, dtype = torch.float32)
        #         get_Q_func_matrix = get_Q_func_matrix + identity_matrix
        #         V_array = torch.ones((1,self.n_actions), dtype = torch.float32)
        #         get_Q_func_matrix = torch.cat((get_Q_func_matrix, V_array), dim = 0)
        #         meta_state = torch.matmul(meta_state, get_Q_func_matrix)

        # state = meta_state
        # return state



class DDPG_NeuronNetwork_Critic(nn.Module):
    ''' neuron network for ddpg critic'''
    
    def __init__(self, n_actions: int, n_inputs = 10, layers = 3, neuronNums= [400,200,200]):
    # def __init__(self, n_actions: int, n_inputs = 10, layers = 3, neuronNums= [400,200,200], dueling = True):
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
        n_outputs = 1

        # # experience buffer variables
        # self.dueling = dueling
        self.linear1 = nn.Linear(n_inputs, neuronNums[0])
        self.act1 = nn.ReLU()
        # if (layers = 3):
        #     raise Exception("layers other than 3 not supported")
        self.linear2 = nn.Linear(neuronNums[0], neuronNums[1])
        self.act2 = nn.ReLU()

        if (layers == 3):
            self.linear3 = nn.Linear(neuronNums[1], neuronNums[2])
            self.act3 = nn.ReLU()
            self.linear4 = nn.Linear(neuronNums[2],n_outputs)
        elif (layers == 2):
            self.linear3 = nn.Linear(neuronNums[1], n_outputs)
        # # extra tanh
        # self.act4 = nn.Tanh()

        
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.to(self.device)

        # # for index in range(layers):
        # #     self.linearLists[index] = nn.Linear(inputCount, neuronNums[index])
        # #     self.actLists[index] = nn.ReLU()
        # #     inputCount = neuronNums[index]
        # #     if (index == layers - 1):
        # #         self.linearLists[index + 1] = nn.Linear(inputCount, n_outputs)
        # self.linear1 = nn.Linear(n_inputs, neuronNums[0])
        # self.act1 = nn.ReLU()
        # if (layers == 2):
        #     self.linear2 = nn.Linear(neuronNums[0], neuronNums[1])
        #     self.act2 = nn.ReLU()
        #     if (dueling):
        #         self.linear3 = nn.Linear(neuronNums[1], n_actions + 1)   # add an extra V(s)
        #         # self.act3 = nn.ReLU()
        #     else:
        #         self.linear3 = nn.Linear(neuronNums[1], n_actions)
        # else:
        #     if (dueling):
        #         raise Exception("Not implemented dueling for single layer yet")
        #     else:
        #         self.linear2 = nn.Linear(neuronNums[0], n_actions)
            

    def forward(self, state_tensor, action_tensor):
        ''' Performs a forward computation '''
        meta_state = torch.cat([state_tensor,action_tensor],1)
        # for index in range(self.layers):
        #     meta_state = self.linearLists[index](meta_state)
        #     meta_state = self.actLists[index](meta_state)
        meta_state = self.linear1(meta_state)
        meta_state = self.act1(meta_state)
        meta_state = self.act2(self.linear2(meta_state))
        if (self.layers == 3):
            meta_state = self.act3(self.linear3(meta_state))
            state = self.linear4(meta_state)
        elif (self.layers == 2):
            state = self.linear3(meta_state)
        # state = self.act4(self.linear4(meta_state))
        return state
        

        # meta_state = state
        # # for index in range(self.layers):
        # #     meta_state = self.linearLists[index](meta_state)
        # #     meta_state = self.actLists[index](meta_state)
        # meta_state = self.linear1(meta_state)
        # meta_state = self.act1(meta_state)
        # meta_state = self.linear2(meta_state)
        # if (self.layers == 2):
        #     meta_state = self.act2(meta_state)
        #     meta_state = self.linear3(meta_state)
        #     if (self.dueling):
        #         # create tensor for column operation
        #         # get_Q_func_matrix = torch.ones((2,), dtype = torch.float32, requires_grad=False)
        #         get_Q_func_matrix = torch.full((self.n_actions, self.n_actions), -1/self.n_actions)
        #         identity_matrix = torch.eye(self.n_actions, dtype = torch.float32)
        #         get_Q_func_matrix = get_Q_func_matrix + identity_matrix
        #         V_array = torch.ones((1,self.n_actions), dtype = torch.float32)
        #         get_Q_func_matrix = torch.cat((get_Q_func_matrix, V_array), dim = 0)
        #         meta_state = torch.matmul(meta_state, get_Q_func_matrix)

        # state = meta_state
        # return state



class DDPG_agent(Agent):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int,
     discount_factor = 0.95, buffer_size = 10000,  tau = 1e-3,
    cer = True,train_batch_size = 128,
    init_fill_fraction = 1.,
    episodes = 1000, target_freq_update = 50, 
    actor_learning_rate = 5e-5, critic_learning_rate = 5e-4, d = 2, n_inputs = 8, 
    actor_neuron_layers = 2, actor_neuronNums= [400,200], 
    critic_neuron_layers = 2, critic_neuronNums = [400,200], 
    mu = 0.15,sigma = 0.2,
    gradient_clip = True, gradient_clip_max = 1., premature_stop = True, threshold = 120, optimal_len = 50,
    adaptiveC=False,
    loadPrev = False, Q_networkFileName = 'neural-network-2-critic.pth',
    pi_networkFileName = 'neural-network-2-actor.pth',
    updateActorUseTarget = False):
        '''

        ----
        Parameters:
        ----

        eps_decay_method : variable to indicate which method for epsilon decay is used
            which will be given to eps_decay_Func : used to calculate the current epsilon

        n_inputs : number of inputs, default 8  (state space dimension)

        buffer_size : size of the buffer

        cer : use  CER for replay buffer, default True
         
        d : freq to update target

        layers:  number of layers

        neurons: list each element is the number of neurons per layer

        gradient_clip : boolean,
            to clip the gradient when doing back propagation, default True
        
        gradient_clip_max : float
            absolute cliping value, default 1.0

        premature_stop : stop if performs good for some time
        
        threshold : default 120
        
        optimal_len = 50        

        '''
        super().__init__(n_actions)
        self.n_actions = n_actions
        self.last_action = None
        self.actor_neuron_layers = actor_neuron_layers
        self.actor_neuronNums = actor_neuronNums
        self.critic_neuron_layers = critic_neuron_layers
        self.critic_neuronNums = critic_neuronNums

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discount_factor = discount_factor

        self.tau = tau

        self.d = d

        # self.linearLists = [None] * (self.layers + 1)
        # self.actLists = [None] * self.layers
        # inputCount = n_inputs
        # n_outputs = n_actions
        
        # main neuron networks for Q
        # self.main_Q_nn = DDPG_NeuronNetwork_Critic(n_actions, n_inputs, layers, neuronNums=neuronNums, dueling=dueling)
        if (not loadPrev):
            self.main_Q_nn = DDPG_NeuronNetwork_Critic(n_actions, n_inputs=10, layers=critic_neuron_layers, neuronNums=critic_neuronNums)
        else:
            self.main_Q_nn = torch.load(Q_networkFileName)


        # target neuron networks for Q
        self.target_Q_nn = copy.deepcopy(self.main_Q_nn)
        self.target_Q_nn.load_state_dict(self.main_Q_nn.state_dict())

        # main neuron networks for actor pi
        if (not loadPrev):
            self.main_pi_nn = DDPG_NeuronNetwork_Actor(n_actions, n_inputs=8, layers=actor_neuron_layers, neuronNums=actor_neuronNums)
        else:
            self.main_pi_nn = torch.load(pi_networkFileName)
        # target neuron networks for Q
        self.target_pi_nn = copy.deepcopy(self.main_pi_nn)
        self.target_pi_nn.load_state_dict(self.main_pi_nn.state_dict())

        # optimiser
        self.main_Q_optimizer = optim.Adam(self.main_Q_nn.parameters(), lr = critic_learning_rate)
        self.main_pi_optimizer = optim.Adam(self.main_pi_nn.parameters(), lr = actor_learning_rate)

        # noise

        self.mu = mu
        self.sigma = sigma

        self.n_t = np.zeros((2,))
        self.w_t = np.zeros((2,))

        # cliping
        self.gradient_clip = gradient_clip
        self.gradient_clip_max = gradient_clip_max
        # loss function
        self.Q_loss_fn = torch.nn.MSELoss()

        # replay buffer

        self.buffer = ExperienceReplayBuffer(maximum_length=buffer_size, CER = cer)
        self.buffer_size = buffer_size
        self.train_batch_size = train_batch_size
        self.target_freq_update = target_freq_update
        self.init_fill_fraction = init_fill_fraction

        # move count

        self.move_count = 0

        # episode count
        self.episode = 0

        # for update using actor using critic target nn
        self.updateActorUseTarget = updateActorUseTarget
        # epsilon calculate function
        # self.eps_decay_Func = epsilonDecay(method = eps_decay_method)
        # self.epsilon = 0

        # for index in range(layers):
        #     self.linearLists[index] = nn.Linear(inputCount, neuronNums[index])
        #     self.actLists[index] = nn.ReLU()
        #     inputCount = neuronNums[index]
        #     if (index == layers - 1):
        #         self.linearLists[index] = nn.Linear(inputCount, n_outputs)
    def reset(self, episode = 0):
        ''' reset parameters
        
        set the move_count variable to 0,

        set the episode to a given variable

        return the current epsilon for eps-greedy
        '''
        self.move_count = 0
        self.episode = episode
        self.n_t = np.zeros((2,))

        # self.epsilon = self.eps_decay_Func(episode)
        # return self.epsilon

    def init_ExperienceBuffer(self, env, method = 1, **kargs):
        ''' initialize experience buffer
        
        --- 
        Parameters
        ----

        env : gym environment

        method : indicate which method to use for init
            1 : means use random, default 50 episode length
        '''
            # initialize buffers
        state = env.reset()

        # total len to be filled
        fillLen = np.int64(self.buffer_size * self.init_fill_fraction)

        if method == 1:
            try :
                updateLen = kargs["updateLen"]
            except KeyError:
                print ("need to indicate updateLen, otherwise choose default 50")
                updateLen = 50
            else:
                    
                for i in range(fillLen):
                    # generate 2-dim action [-1,1]
                    # action = np.random.randint(0,self.n_actions)
                    action = np.random.rand(2)* 2 - 1
                    next_state, reward, done, _ = env.step(action)
                    # experience created
                    exp = Experience(state, action, reward, next_state, done)
                    self.buffer.append(exp)
                    state = next_state
                    if (i % 50 == 0) or done:
                        state = env.reset()

    def takeAction(self, state_tensor):
    # def takeAction(self, epsilon, state_tensor):
        ''' take the action'''

        # action = self.main_Q_nn(state_tensor, action_tensor)
        action_tensor = self.main_pi_nn(state_tensor)
        w = np.random.normal(self.mu, self.sigma, (2,))
        self.n_t = - self.mu * self.n_t + w
        action = action_tensor.data.numpy() + self.n_t
        # # Take a random action, epsilon greedy
        # if (random.random() < epsilon):
        #     action = np.random.randint(0, self.n_actions)
        # else :
        #     action = np.argmax(qval.data.numpy())

        return action

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''

        state_tensor = torch.tensor(state, requires_grad=False, dtype=torch.float32)
        # calculate action using main actor
        # action_tensor = self.main_pi_nn(state_tensor)
        # calculate epsilon
        # epsilon = eps_decay_method(self.episode)
        action = self.takeAction(state_tensor)
        self.move_count += 1
        # self.episode += 1
        return action
    
    def appendExperience(self, state, action, reward, next_state, done):
        ''' append newly acquired experience'''
        # experience created
        exp = Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)

    def backward(self):
        ''' Performs a backward pass on the network '''
        # target Value
        # ..
        # loss
        # optimitiyer.yero grad
        # loss.backward
        # optimiyer.step

        # mini batch train NN
        if (len(self.buffer) >= self.train_batch_size):
            # sample
            states, actions, rewards, next_states, dones = self.buffer.sample_batch(n = self.train_batch_size)

            # first convert states and actions to np.array
            states = np.array(states)
            states_tensor = torch.tensor(states, requires_grad=True, dtype=torch.float32)
            actions = np.array(actions)
            next_states = np.array(next_states)
            next_states_tensor = torch.tensor(next_states, requires_grad=False,dtype=torch.float32)

            # training process, set grad to 0
            self.main_Q_optimizer.zero_grad()

            # get value
            main_Q_val = self.main_Q_nn(states_tensor, torch.tensor(actions, requires_grad=True,dtype=torch.float32))
            main_Q_val = main_Q_val.squeeze()
            # do not need grad
            with torch.no_grad():
                action_target_tensors = self.target_pi_nn(next_states_tensor)
                Q2 = self.target_Q_nn(next_states_tensor, action_target_tensors)
                Q2 = Q2.squeeze()
            rewards = torch.tensor(rewards, requires_grad=False,dtype=torch.float32)
            dones = torch.tensor(dones, requires_grad=False,dtype=torch.float32)

            target_Q_val = rewards + self.discount_factor * (1 - dones) * Q2
            # calculated using updated model
            # oldRewards = main_Q_val.gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
            loss = self.Q_loss_fn(main_Q_val, target_Q_val.detach())

            loss.backward()
            # debug check grad
            if (PRINT_GRAD):
                for param in self.main_Q_nn.parameters():
                    print(param.grad)            
            # clip gradient
            if (self.gradient_clip):
                nn.utils.clip_grad_norm_(self.main_Q_nn.parameters(), max_norm=self.gradient_clip_max)
            self.main_Q_optimizer.step()
            # update the agent target
            if self.move_count % self.d == 0:
                # states_tensor = torch.tensor(states, requires_grad=True,
                #                     dtype=torch.float32)
                # check if requires_grad == True
                action_tensor = self.main_pi_nn(states_tensor)
                if (self.updateActorUseTarget):
                    policy_loss = -self.target_Q_nn(states_tensor.detach(), action_tensor).mean() 
                else:
                    policy_loss = -self.main_Q_nn(states_tensor.detach(), action_tensor).mean() 
                # # or
                # policy_loss = -self.main_Q_nn(states_tensor.detach(), action_tensor).detach().mean()

                self.main_pi_optimizer.zero_grad()
                policy_loss.backward()
                if (PRINT_GRAD):
                    for param in self.main_pi_nn.parameters():
                        print(param.grad)
                if (self.gradient_clip):
                    nn.utils.clip_grad_norm_(self.main_pi_nn.parameters(), max_norm=self.gradient_clip_max)
                    
                self.main_pi_optimizer.step()
                self.target_pi_nn = soft_updates(self.main_pi_nn, self.target_pi_nn, self.tau)
                self.target_Q_nn = soft_updates(self.main_Q_nn, self.target_Q_nn, self.tau)

                # self.target_nn.load_state_dict(self.main_nn.state_dict())
    def save_main_Q_nn(self, fileName):
        ''' save the main Q nn parameters'''
        torch.save(self.main_Q_nn, fileName)

    def save_main_pi_nn(self, fileName):
        ''' save the main pi nn parameters'''
        torch.save(self.main_pi_nn, fileName)
