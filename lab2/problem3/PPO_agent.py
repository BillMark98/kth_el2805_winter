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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 29th October 2020, by alessior@kth.se
#

# Load packages
from math import gamma
import numpy as np
import torch
from torch.distributions import MultivariateNormal
# from scipy.stats import multivariate_normal

from torch.autograd import grad
# torch neuron network
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
import random

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

PRINT_GRAD = False
# PRINT_GRAD = True


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
    def clear(self):
        ''' clear buffer '''
        self.buffer.clear()

    def getStates(self):
        ''' get the states vector '''
        states_vec = [exp.state for exp in self.buffer]
        return states_vec

    def getActions(self):
        actions_vec =  [exp.action for exp in self.buffer]
        return actions_vec
    def getRewards(self):
        rewards_vec = [exp.reward for exp in self.buffer]
        return rewards_vec
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
                    the parent class Agent
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)




class PPO_NeuronNetwork_Actor(nn.Module):
    ''' neuron network for PPO actor'''
    
    def __init__(self, n_actions: int, n_inputs = 8, layers = 2, neuronNums= [400,200]):
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
        # self.linear2 = nn.Linear(neuronNums[0], neuronNums[1])
        # self.act2 = nn.ReLU()
        if (layers == 3):
            raise Exception("currently 3 layers not supported")
            self.linear3 = nn.Linear(neuronNums[1], neuronNums[2])
            self.act3 = nn.ReLU()
            self.linear4 = nn.Linear(neuronNums[2],n_outputs)
            # extra tanh
            self.act4 = nn.Tanh()
        elif (layers == 2):
            # mu hidden
            self.linear2_mu = nn.Linear(neuronNums[0], neuronNums[1])
            self.act2_mu = nn.ReLU()
            # sigma hidden
            self.linear2_sigma = nn.Linear(neuronNums[0], neuronNums[1])
            self.act2_sigma = nn.ReLU()

            # mu output
            self.linear3_mu = nn.Linear(neuronNums[1], n_outputs)
            self.act3_mu = nn.Tanh()

            # sigma output
            self.linear3_sigma = nn.Linear(neuronNums[1], n_outputs)
            self.act3_sigma = nn.Sigmoid()

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
            

    def forward(self, state_tensor):
        ''' Performs a forward computation '''
        meta_state = state_tensor
        # for index in range(self.layers):
        #     meta_state = self.linearLists[index](meta_state)
        #     meta_state = self.actLists[index](meta_state)
        meta_state = self.linear1(meta_state)
        meta_state = self.act1(meta_state)
        # meta_state = self.act2(self.linear2(meta_state))
        # meta_state = self.act3(self.linear3(meta_state))
        if (self.layers == 3):
            raise Exception("layer 3 not supported in actor")
            meta_state = self.act4(self.linear4(meta_state))
        elif (self.layers == 2):
            meta_mu = self.linear2_mu(meta_state)
            meta_sigma = self.linear2_sigma(meta_state)
            meta_mu = self.act2_mu(meta_mu)
            meta_sigma = self.act2_sigma(meta_sigma)

            # output
            meta_mu = self.act3_mu(self.linear3_mu(meta_mu))
            meta_sigma = self.act3_sigma(self.linear3_sigma(meta_sigma))
            

        state_tensor = torch.cat([meta_mu, meta_sigma], -1)
        return state_tensor
        
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



class PPO_NeuronNetwork_Critic(nn.Module):
    ''' neuron network for PPO critic'''
    
    def __init__(self, n_actions: int, n_inputs = 8, layers = 2, neuronNums= [400,200]):
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
            

    def forward(self, state_tensor):
        ''' Performs a forward computation '''
        # meta_state = torch.cat([state_tensor,action_tensor],1)
        meta_state = state_tensor
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



class PPO_agent(Agent):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int,
     discount_factor = 0.95, buffer_size = 10000,
    episodes = 1000,  
    M = 10, epsilon = 0.2,
    actor_learning_rate = 5e-5, critic_learning_rate = 5e-4,
    n_inputs = 8, 
    actor_neuron_layers = 2, actor_neuronNums= [400,200], 
    critic_neuron_layers = 2, critic_neuronNums = [400,200], 
    gradient_clip = True, gradient_clip_max = 1., premature_stop = True, 
    threshold = 120, optimal_len = 50,
    loadPrev = False, V_networkFileName = 'neural-network-3-critic.pth',
    pi_networkFileName = 'neural-network-3-actor.pth'):
        '''

        ----
        Parameters:
        ----

        eps_decay_method : variable to indicate which method for epsilon decay is used
            which will be given to eps_decay_Func : used to calculate the current epsilon

        n_inputs : number of inputs, default 8  (state space dimension)

        buffer_size : size of the buffer

        M : period for training

        epsilon : the epsilon in the J_ppo 

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
        self.accumulate_gamma = 1.




        # self.linearLists = [None] * (self.layers + 1)
        # self.actLists = [None] * self.layers
        # inputCount = n_inputs
        # n_outputs = n_actions
        
        # main neuron networks for V
        # self.main_V_nn = PPO_NeuronNetwork_Critic(n_actions, n_inputs, layers, neuronNums=neuronNums, dueling=dueling)
        if (not loadPrev):
            self.main_V_nn = PPO_NeuronNetwork_Critic(n_actions, n_inputs=8, layers=critic_neuron_layers, neuronNums=critic_neuronNums)
        else:
            self.main_V_nn = torch.load(V_networkFileName)
        
        # # target neuron networks for V
        # self.target_V_nn = copy.deepcopy(self.main_V_nn)
        # self.target_V_nn.load_state_dict(self.main_V_nn.state_dict())

        # main neuron networks for actor pi
        if (not loadPrev):
            self.main_pi_nn = PPO_NeuronNetwork_Actor(n_actions, n_inputs=8, layers=actor_neuron_layers, neuronNums=actor_neuronNums)
        else:
            self.main_pi_nn = torch.load(pi_networkFileName)

        # # target neuron networks for pi
        # self.target_pi_nn = copy.deepcopy(self.main_pi_nn)
        # self.target_pi_nn.load_state_dict(self.main_pi_nn.state_dict())

        # optimiser
        self.main_V_optimizer = optim.Adam(self.main_V_nn.parameters(), lr = critic_learning_rate)
        self.main_pi_optimizer = optim.Adam(self.main_pi_nn.parameters(), lr = actor_learning_rate)

        # # noise

        # self.mu = mu
        # self.sigma = sigma

        # self.n_t = np.zeros((2,))
        # self.w_t = np.zeros((2,))

        # cliping
        self.gradient_clip = gradient_clip
        self.gradient_clip_max = gradient_clip_max
        # loss function
        self.V_loss_fn = torch.nn.MSELoss()

        # replay buffer

        self.buffer = ExperienceReplayBuffer(maximum_length=buffer_size)
        self.buffer_size = buffer_size
        self.train_period = M
        self.epsilon = epsilon
        # self.train_batch_size = train_batch_size
        # self.target_freq_update = target_freq_update


        # move count

        self.move_count = 0

        # episode count
        self.episode = 0
        # episode reward
        self.episode_reward = 0.
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
        # clear the experience buffer

        self.buffer.clear()
        self.episode_reward = 0.
        self.accumulate_gamma = 1.
        # self.n_t = np.zeros((2,))

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

        if method == 1:
            try :
                updateLen = kargs["updateLen"]
            except KeyError:
                print ("need to indicate updateLen, otherwise choose default 50")
                updateLen = 50
            else:
                    
                for i in range(self.buffer_size):
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

        # action = self.main_V_nn(state_tensor, action_tensor)
        # get the parameter
        # param_tensor = self.main_pi_nn(state_tensor)
        with torch.no_grad():
            param = self.main_pi_nn(state_tensor)

        # param = param_tensor.data().numpy().squeeze()

        # sample from normal distribution
        # get the mu
        mu_vec = param[0:2].clone()
        sigma_vec = param[2:4].clone()
        action = MultivariateNormal(mu_vec, torch.diag(sigma_vec)).sample()
        # action = np.random.multivariate_normal(mu_vec, np.diag(sigma_vec), 1)
        # w = np.random.normal(self.mu, self.sigma, (2,))
        # self.n_t = - self.mu * self.n_t + w
        # action = action_tensor.data.numpy() + self.n_t
        # # Take a random action, epsilon greedy
        # if (random.random() < epsilon):
        #     action = np.random.randint(0, self.n_actions)
        # else :
        #     action = np.argmax(qval.data.numpy())

        return action

    def getProb(self, state:np.ndarray, action:np.ndarray, requires_grad = False):
        ''' get the probability of take action at state'''
        
        # # convert to torch
        # state_tensor = torch.tensor(state, requires_grad=requires_grad, dtype=torch.float32)
        # # get param
        # param = self.main_pi_nn(state_tensor)
        # # param = param_tensor.data().numpy().squeeze()
        # # mu
        # mu_vec = param[0:2]
        # sigma_vec = param[2:4]
        # # evaluate the probability
        # # pdf = multivariate_normal.pdf(action, mean=mu_vec, cov=np.diag(sigma_vec))
        # action_tensor = torch.tensor(action, requires_grad=requires_grad, dtype=torch.float32)
        # pdf = MultivariateNormal(mu_vec, torch.diag(sigma_vec)).log_prob(action_tensor).exp()
        raise Exception("Should not be called")
        # return pdf
    
    def getProbs(self, states_vec, actions_vec, requires_grad = False):
        ''' given state vectors, action vectors, calculate the probability vectors '''
        # prob_vec = torch.zeros((len(states_vec),))
        # for index in range(len(actions_vec)):
        #     prob_vec[index] = self.getProb(states_vec[index], actions_vec[index], requires_grad)
        states_tensor = torch.tensor(states_vec, requires_grad=requires_grad, dtype=torch.float32)
        # get param
        param = self.main_pi_nn(states_tensor)        
        dist = MultivariateNormal(param[:,0:2].clone(), torch.diag_embed(param[:,2:4].clone()))
        return dist.log_prob(torch.tensor(actions_vec, requires_grad=requires_grad)).exp()

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
        return action.numpy()
    
    def appendExperience(self, state, action, reward, next_state, done):
        ''' append newly acquired experience'''
        # experience created
        exp = Experience(state, action, reward, next_state, done)
        self.buffer.append(exp)
        self.episode_reward += reward * self.accumulate_gamma
        self.accumulate_gamma *= self.discount_factor
    
    def compute_rwtg(self, rewards_vec: np.ndarray):
        ''' compute the reward to go 
        
            return: G_t tensor form
        '''
        # rewards_to_go = np.array(len(rewards_vec))
        def create_gammaMatrix(rowCount=len(rewards_vec), gamma = self.discount_factor):
            ''' create n * n gamma Matrix for the calculation'''
            # def mappingRule(i,j, factor):
            #     if (i < j):
            #         return 0
            #     else:
            #         return np.factor()
            gamma_matrix = np.ones((rowCount))
            gamma_matrix[np.tril_indices(rowCount,-1)] = 0
            for i in reversed(range(1, rowCount)):
                gamma_matrix[np.triu_indices(rowCount, i)] *= gamma
            return gamma_matrix
        
        # for loop version
        rewards_to_go = np.zeros((len(rewards_vec),))
        accumulated_reward = 0
        for i in reversed(range(len(rewards_vec))):
            rewards_to_go[i] = rewards_vec[i] + self.discount_factor * accumulated_reward
            accumulated_reward = rewards_to_go[i]
        
        # gamma_matrix = create_gammaMatrix()
        # rewards_to_go = np.dot(gamma_matrix, rewards_vec)
        return torch.tensor(rewards_to_go,requires_grad=False, dtype=torch.float32)
    def c_eps(self, x):
        ''' the help function for calculating J_ppo '''
        # return np.maximum(1- self.epsilon, np.minimum(x, 1 + self.epsilon))
        return torch.clamp(x, 1 - self.epsilon, 1 + self.epsilon)

    def backward(self):
        ''' Performs a backward pass on the network '''
        # target Value
        # ..
        # loss
        # optimitiyer.yero grad
        # loss.backward
        # optimiyer.step

        # mini batch train NN

        # calculate the psi_old
        # get the state vector
        states_vec = self.buffer.getStates()
        psi_old = self.main_V_nn(torch.tensor(np.array(states_vec), requires_grad=False, dtype=torch.float32))
        rewards_vec = self.buffer.getRewards()
        rewards_to_go_tensor = self.compute_rwtg(rewards_vec)
        psi_old_tensor = rewards_to_go_tensor - psi_old.squeeze()
        action_vec = self.buffer.getActions()
        # calculate the probability
        old_prob_tensor = self.getProbs(states_vec, action_vec, requires_grad=False)
        # for index in range(len(action_vec)):
        #     old_prob_tensor[index] = self.getProb(states_vec[index], action_vec[index])

        states_vec_tensor = torch.tensor(np.array(states_vec), requires_grad=True, dtype=torch.float32)
        # the reference
        for count in range(self.train_period) :
            self.main_V_optimizer.zero_grad()
            new_stateVals = self.main_V_nn(states_vec_tensor).squeeze()
            loss = self.V_loss_fn(rewards_to_go_tensor.detach(), new_stateVals)
            # back ward propagate
            loss.backward()
                        # debug check grad
            if (PRINT_GRAD):
                for param in self.main_V_nn.parameters():
                    print(param.grad)       
            self.main_V_optimizer.step()

            # actor backward
            self.main_pi_optimizer.zero_grad()
            # new prob vec
            new_prob_tensor = self.getProbs(states_vec, action_vec, requires_grad=True)
            # calculate the r_tensor
            r_tensor = torch.div(new_prob_tensor, old_prob_tensor.detach())
            # policy_loss = -1 * torch.minimum(torch.mul(r_tensor, psi_old_tensor), self.c_eps(torch.mul(r_tensor, psi_old_tensor))).mean()
            policy_loss = -1 * torch.minimum(torch.mul(r_tensor, psi_old_tensor.detach()), torch.mul(self.c_eps(r_tensor), psi_old_tensor.detach())).mean()
            policy_loss.backward()
                        # debug check grad
            if (PRINT_GRAD):
                for param in self.main_pi_nn.parameters():
                    print(param.grad)       
            self.main_pi_optimizer.step()

        # clear buffer
        self.buffer.clear()

        # if (len(self.buffer) >= self.train_batch_size):
        #     # sample
        #     states, actions, rewards, next_states, dones = self.buffer.sample_batch(n = self.train_batch_size)

        #     # first convert states and actions to np.array
        #     states = np.array(states)
        #     states_tensor = torch.tensor(states, requires_grad=True, dtype=torch.float32)
        #     actions = np.array(actions)
        #     next_states = np.array(next_states)
        #     next_states_tensor = torch.tensor(next_states, requires_grad=False,dtype=torch.float32)

        #     # training process, set grad to 0
        #     self.main_V_optimizer.zero_grad()

        #     # get value
        #     Q1 = self.main_V_nn(states_tensor, torch.tensor(actions, requires_grad=True,dtype=torch.float32))
        #     Q1 = Q1.squeeze()
        #     # do not need grad
        #     with torch.no_grad():
        #         action_target_tensors = self.target_pi_nn(next_states_tensor)
        #         Q2 = self.target_V_nn(next_states_tensor, action_target_tensors)
        #         Q2 = Q2.squeeze()
        #     rewards = torch.tensor(rewards, requires_grad=False,dtype=torch.float32)
        #     dones = torch.tensor(dones, requires_grad=False,dtype=torch.float32)

        #     new_stateVals = rewards + self.discount_factor * (1 - dones) * Q2
        #     # calculated using updated model
        #     # oldRewards = Q1.gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        #     oldRewards = Q1

        #     loss.backward()
        #     # debug check grad
        #     if (PRINT_GRAD):
        #         for param in self.main_V_nn.parameters():
        #             print(param.grad)            
        #     # clip gradient
        #     if (self.gradient_clip):
        #         nn.utils.clip_grad_norm_(self.main_V_nn.parameters(), max_norm=self.gradient_clip_max)
        #     self.main_V_optimizer.step()
        #     # update the agent target
        #     if self.move_count % self.d == 0:
        #         # states_tensor = torch.tensor(states, requires_grad=True,
        #         #                     dtype=torch.float32)
        #         # check if requires_grad == True
        #         action_tensor = self.main_pi_nn(states_tensor)
        #         policy_loss = -self.main_V_nn(states_tensor.detach(), action_tensor).mean() 
        #         # # or
        #         # policy_loss = -self.main_V_nn(states_tensor.detach(), action_tensor).detach().mean()

        #         self.main_pi_optimizer.zero_grad()
        #         policy_loss.backward()
        #         if (PRINT_GRAD):
        #             for param in self.main_pi_nn.parameters():
        #                 print(param.grad)
        #         if (self.gradient_clip):
        #             nn.utils.clip_grad_norm_(self.main_pi_nn.parameters(), max_norm=self.gradient_clip_max)
                    
        #         self.main_pi_optimizer.step()
        #         self.target_pi_nn = soft_updates(self.main_pi_nn, self.target_pi_nn, self.tau)
        #         self.target_V_nn = soft_updates(self.main_V_nn, self.target_V_nn, self.tau)

        #         # self.target_nn.load_state_dict(self.main_nn.state_dict())
    def save_main_V_nn(self, fileName):
        ''' save the main Q nn parameters'''
        torch.save(self.main_V_nn, fileName)

    def save_main_pi_nn(self, fileName):
        ''' save the main pi nn parameters'''
        torch.save(self.main_pi_nn, fileName)


if __name__ == "__main__":
    test_ppo_agent = PPO_agent(n_actions=8)
