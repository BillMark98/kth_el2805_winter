
import os
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))



# load the parameters

# Load model
try:
    model = torch.load('neural-network-1.pth')
    print('Network model: {}'.format(model))
except:
    print('File neural-network-1.pth not found!')
    exit(-1)


# Import and initialize Mountain Car Environment
env = gym.make('LunarLander-v2')
env.reset()

y_Nums = 100
omega_Nums = 100

# color map scheme
value_func_cmap = cm.coolwarm
# action_cmap = cm.coolwarm
action_cmap = "binary"


# save Name
valueFuncSaveName = "value_function_plt.png"
actionSaveName = "action_plt.png"

# create Y 
y = np.linspace(0,1.5, y_Nums)
omega = np.linspace(-np.pi, np.pi, omega_Nums)

X_coords, Y_coords = np.meshgrid(y, omega)

# flatten 
X_coords_flatten = np.ravel(X_coords)
Y_coords_flatten = np.ravel(Y_coords)
states = np.array([[0,y,0,0,omega,0,0,0] for y, omega in zip(X_coords_flatten, Y_coords_flatten)])

# calculate q_function
q_values = model(torch.tensor(states, dtype=torch.float32))

max_q, action = torch.max(q_values, axis = 1)
max_q = max_q.data.numpy().reshape(X_coords.shape)
action = action.data.numpy().reshape(X_coords.shape).astype('float32')
# print(states)
# print(max_q)
# print(action)
fig= plt.figure()
ax = plt.axes(projection='3d')
# ax.contour3D(X_coords, Y_coords, max_q, 50, cmap=value_func_cmap)
ax.plot_surface(X_coords, Y_coords, max_q, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('y')
ax.set_ylabel('omega')
ax.set_zlabel('value function');
print("value function saved at " + valueFuncSaveName)
plt.savefig(valueFuncSaveName)
# plt.show()


fig= plt.figure()
ax = plt.axes(projection='3d')
# ax.contour3D(X_coords, Y_coords, action, 50, cmap=action_cmap)
# ax.plot_wireframe(X_coords, Y_coords, action, color='black')
ax.plot_surface(X_coords, Y_coords, action, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('y')
ax.set_ylabel('omega')
ax.set_zlabel('action');
print("action saved at " + actionSaveName)
plt.savefig(actionSaveName)
plt.show()

# def calculate_single_q(y,omega):
#     ''' given a single y, omega calculate the max Q value '''
#     state = np.array([0,y,0,0,omega,0,0,0])
#     q_values = model(torch.tensor([state]))
