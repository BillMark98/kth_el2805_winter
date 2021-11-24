# python notes

could not use list as key for the dictionary, [reason](https://www.geeksforgeeks.org/how-to-use-a-list-as-a-key-of-a-dictionary-in-python-3/)

because list is mutable
could convert to string, or tuple

* [select indices](https://stackoverflow.com/questions/3030480/how-do-i-select-elements-of-an-array-given-condition)

## generate gif

* [draw gif](https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/)

* [good tutorial](https://linuxtut.com/en/3c66781f41884694838b/)

In essence, the idea of `FuncAnimation` is

```python
for d in frames:
   artists = func(d, *fargs)
   fig.canvas.draw_idle()
   fig.canvas.start_event_loop(interval)
```

so example

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(111)

def update(frame):
    ax.cla() #Clear ax
    ax.plot(frame, 0, "o")

anim = FuncAnimation(fig, update, frames=range(8), interval=1000)

anim.save("c02.gif", writer="imagemagick")
plt.close()
```

In particular `update` does not need to return a function, in this case of `maze`, we do not need to return something, so do the plot stuff within that function,  note that the scope of the required `grid` and `fig` variable. If defined in a separte function, say `make_updateFunc` then the `grid` and `fig` is actually lost, so when calling, will generate some errors. So the solution is to define the function within the `animate_simulation` to ensure the scope of variables are valid.

* [maybe interesting](https://stackoverflow.com/questions/37111571/how-to-pass-arguments-to-animation-funcanimation), another [implementation for maze](https://github.com/timothygordon32/mazes/blob/master/maze.py), which use `pyplot.imshow` which displays a image based on numerical array (numpy.array, 2d) guess value is interpretated as gray value

# jupyter lab

* [debug](https://www.youtube.com/watch?v=DN8lGUI72Ms)


# git notes

* delete a remote branch

```git
git push origin --delete <branch>
```

* when try to push to a remote branch, syntax:

```git

git push remoteRepoName localBranch:remoteBranch
```


# tex notes

* [tilde over letter](https://yoo2080.wordpress.com/2014/11/30/putting-a-bar-or-a-tilde-over-a-letter-in-latex/#:~:text=To%20put%20a%20tilde%20over,its%20part%20can%20help%20comparison.&text=To%20put%20a%20bar%20over,%5Ctilde%20is%20to%20%5Cwidetilde%20.)

* [default argument](https://stackoverflow.com/questions/1812214/latex-optional-arguments), [another link](https://tex.stackexchange.com/questions/58628/optional-argument-for-newcommand), [more than one optional arguments](https://tex.stackexchange.com/questions/29973/more-than-one-optional-argument-for-newcommand)

* [mathmode xparse](https://tex.stackexchange.com/questions/472981/math-mode-parsing-in-newdocumentcommand)

* [xparse docu](https://ftp.acc.umu.se/mirror/CTAN/macros/latex/contrib/l3packages/xparse.pdf)

# problem 1

## MDP

### reward

reward divides into two types,

the first one is related to the (s,a) 

the second is only related to the state being eaten

```python
                    next_s = self.__move(s,a);
                    # Reward for being eaten :-<
                    if self.__isEaten(s):
                        reward[s,a] = self.EATEN_REWARD;
```

### transition

*  if exit, then always stay in that status
* if eaten, always stay in that status, in particular could not move to a normal state by some actions

* note the configuration of the matrix
it is 

(next_state, current_state, action), in particular when fixing one action the resulting is a transition_matrix in the MC context, but transposed!

if M = transition_prob(:,:, a)
so M(i,j) = P(i | j)  = "prob reaching i starting at j"
so given a probability state vector, the next step state distribution is M * v  not v^{t} * M

# variables

* rewards :  n_states * n_actions

* Q : n_states * n_actions

# algo

## dp

### variables

* policy : n_states * (T + 1)
* V : n_states * (T + 1)


## value iteration

* the provided template uses the end criterium `while s != next_s` guess it is because the only time that the agent will stay is when he is at the exit in the lab0 maze, but now due to minotaur, it is possible that the agent chooses to stay for a while during the expedition. So uses another criterion, namely whether or not the state is a terminal state

## difference between dp and vi

* the dp is given a time horizon, where as vi not
* the end critirium for dp is the reach of time horizon, whereas vi is the reach of a terminal state
* dp policy could be time dependent,  while as vi is stationary(does not depend on time), so the policy returned from dp is  `n_states * T` , the policy returned from vi is `n_states * 1` 

## q-learning

* [good example](https://www.analyticsvidhya.com/blog/2021/04/q-learning-algorithm-with-step-by-step-implementation-using-python/)

# key picking

add one coordinate to indicate if key already picked, in this case
there might be several states that indicate key exactly picked (due to the variable status of the minotaur), these will have a direct transition to the state with exactly the same position, with the picked (5-th) coordinate changed to true(1), this is the only case where the key-not-picked universe could transfer to the key-picked universe  (what is implemented is that for key picking positions, the next position is directly with key-picked set to true, so in essence, there are no (key_picking_pos,keyPicked = 1))

## question

* will it cause problem if after some time, revisit these states (so key already picked)

# Todo

* how to incoporate the situation

being eaten and reach the exit   for example reward?

* should the `isExit` contain the test of `isEaten?`

* at the `__move` function, what to do if move is hitting wall, should
the minatour still make a random move, and the next_state is the current positioin of the agent x randomly moved position of the minatour?

* for the state of `exit` define the optimal action is stay,

so only the stay will give optimal 0 reward, other rewards is -1

* in `__move`

do we need to check if next state is exit to adjust the reward?

```python
            for index in range(len(next_statePos)):
                # check if being eaten
                if (self.__isEaten(next_statePos[index])):
                    next_statePos[index] = self.STATE_EATEN
                    next_stateNum[index] = self.STATE_EATEN
                    rewards[index] = self.EATEN_REWARD
                # # check if at exit
                # elif (self.__isExit(next_statePos[index])):
                #     next_statePos[index] = self.STATE_EXIT
                #     next_stateNum[index] = self.STATE_EXIT
                #     rewards[index] = self.GOAL_REWARD
                else:
                    next_stateNum[index] = self.map[next_statePos[index]]

```

or simply if a valid state (not eaten), then take -1 reward for a normal step and let the next state (being exit) decide further rewards

guess this is the right way

* how to define the state when being in exit, and make a non-stay move

current version:
still defines it to be in state stay,  but will cost a usual step (-1), so will force the agent to stay as long as it hits exit status

* how to represent maze

for example, cannot define state by a single value,
because exit and the position of minatour can overlap

* depend on whether time critical (get as quickly as possible to exit), or safe critical (be safe from minatour as quickly as possible), could use different reward,

(in abs)
e.g. if time critical,  then the step_reward would be larger,
if safety critial, then the eaten_reward larger 

* for graphical display

need to give explicit what the state is, if at exit, 
or be eaten

if eaten, then will only be a minotaur,

if exit, then will also only display minotaur (the agent is out of the maze)

*

```bash
$ find . -type f \( -name '*.md' -o -name '*.py' -name '*.ipynb' \) -print0 | xargs -0 -I  {} git add {} --dry-run
```

seems not working only add markdown!!!

* optional argument for \expect like \expect_{\pi}

* V_t or V^{\pi}(t)

# Question

* why?
    tol = (1 - gamma)* epsilon/gamma;
* 

```tex
\NewDocumentCommand{\vPiTS}{ O{$\pi$} O{t} O{s} }{V^{#1}{#2}(#3)}
```

seems not working ...
