import numpy as np

policy1 = np.load("policy1.npy")
policy2 = np.load("policy2.npy")
states1 = np.load("states1.npy", allow_pickle=True)
states2 = np.load("states2.npy", allow_pickle = True)
map1 = np.load("map1.npy", allow_pickle = True)
map2 = np.load("map2.npy", allow_pickle = True)

#first print out mikaels s-(i,j,k,l) pair
#then print out panweis s-(i,j,k,l) pair
print(states1)
print("*****************************************************************************")
print("*****************************************************************************")
print("*****************************************************************************")
print(states2)
wrongs1 = []
wrongs = []

rights = []
for s in range(np.shape(policy1)[0]):

    #convert s to (i,j,k,l)
    states_s = states1.item()[s]
    if states_s in map2.item():
        #do something
        s_2 = map2.item()[states_s]
        if (policy1[s] == policy2[s_2]).all():
            rights.append(s_2)
        else:
            wrongs.append(s_2)
            wrongs1.append(s)


print("Total number of states where the policy is different: ",len(wrongs))
print("Total number of states where the policy is the same: ",len(rights))

#print out the policies for the different
for k in range(len(wrongs)):
    print("--------------------------------------------")
    print("Mikael's state: ", wrongs1[k])
    print(policy1[wrongs1[k]])
    print("Panwei's state: ", wrongs[k])
    print(policy2[wrongs[k]])
