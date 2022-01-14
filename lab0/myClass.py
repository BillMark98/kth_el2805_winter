from collections import deque
import random
import torch
import torch.nn as nn

class ExperienceReplayBuffer(deque):

    def __init__(self, maxLen):
        super().__init__(maxlen=maxLen)
    
    def randomChoose(self, N):
        return random.sample(self, N)

class p3Neuron(nn.Module):
    
    def __init__(self,inputDim, outputDim, neuronNum = 8):
        super().__init__()
        self.linear1 = nn.Linear(inputDim, neuronNum)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(neuronNum, outputDim)

    def forward(self, input):
        x = self.linear1(input)
        x = self.act1(x)
        x = self.linear2(x)

        return x
        
if __name__ == "__main__":
    testBf = ExperienceReplayBuffer(5)
    testBf.append(1)
    testBf.append(2)
    testBf.append(30)
    testBf.append(32)
    testBf.append(31)
    print(testBf)

    testBf.append(5)
    print(testBf)

    rR = testBf.randomChoose(4)

    print(rR)
