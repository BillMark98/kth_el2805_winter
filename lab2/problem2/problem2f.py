# Mikael Westlund   personal no. 9803217851
# Panwei Hu t-no. 980709T518
#  
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
    critic_nn = torch.load('neural-network-2-critic.pth')
    actor_nn = torch.load('neural-network-2-actor.pth')
    print('Network model: {}'.format(critic_nn))
    print('Network model: {}'.format(actor_nn))

except:
    print('File neural-network-1.pth not found!')
    exit(-1)


# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
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

states_tensor = torch.tensor(states, dtype=torch.float32)
# calculate q_function
action_tensors = actor_nn(states_tensor)
q_values = critic_nn(states_tensor, action_tensors)

max_q = q_values.data.numpy().reshape(X_coords.shape)
action_coord2 = action_tensors[:,1].data.numpy().reshape(X_coords.shape).astype('float32')
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
plt.title("state action function")
print("value function saved at " + valueFuncSaveName)
plt.savefig(valueFuncSaveName)
# plt.show()


fig= plt.figure()
ax = plt.axes(projection='3d')
# ax.contour3D(X_coords, Y_coords, action, 50, cmap=action_cmap)
# ax.plot_wireframe(X_coords, Y_coords, action, color='black')
ax.plot_surface(X_coords, Y_coords, action_coord2, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('y')
ax.set_ylabel('omega')
ax.set_zlabel('action coord2');
plt.title("action for direction")
print("action saved at " + actionSaveName)
plt.savefig(actionSaveName)
plt.show()

# def calculate_single_q(y,omega):
#     ''' given a single y, omega calculate the max Q value '''
#     state = np.array([0,y,0,0,omega,0,0,0])
#     q_values = model(torch.tensor([state]))
