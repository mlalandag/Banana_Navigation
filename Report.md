
# Report

[image1]: scores.jpg

## Learning Algorithm

In order to solve the Unity Banana Navigation environment for this project we use a neural network (actually a couple of them)
implementing what is call a DQN or Deep Q Network algorithm. Neural Networks are good function aproximators and the idea is to 
use them to approximate the Q-function and, using the state of the environment as input, deliver the values asociated to each 
action so we can pick up the best of them. The model of the chosen topology for the neural network is defined in the model.py
file using the pytorch framework and is described as follows:  
  
### Model

* The Neural Network has two Hidden layers with 64 neurons each.
* The activation function used is `ReLU`
* The output layer has as many neurons as number of possible actions the agent can choose, which is 4

This model is used by the agent (defined in the agent.py file) so it can pick and action from each state of the environment
it is presented to it. The chosen action is then sent to the environment which in turn returns the next state, the reward obtained
and if the episode has concluded or not. Based in the obtained rewards, the neural network weights are actualized with and optimization
algorithm. 

However the information obtained by the agent is generally not identically distributed, being each state strongly correlated 
with the inmediately previous ones. This has to be with the tipically sequential nature of the problems meant to be solved with
these kind of algorithms. This is why a technique called Experience replay is used to obtain samples of the environment past states,
actions, rewards and next states tuples that are not correlated with each other. Experience replay (implemented in the replay_buffer.py file and used by the agent_step method) 
also permits reusing past experience data and it consists in keeping track of the tuples (state, action,reward, next state, done)
encountered by the agent in a buffer and, after a number of steps, sample from it some of them in a way that they are not correlated.
These samples are then used to optimize the neural network in the agent´s "learn" method.

Actions are chosen following an epsilon-greedy policy which takes the best action based in the agent´s action-value estimate with a 
probability of (1 - epsilon) and a random action with a probability given by epsilon. With a higher value of epsilon the agent will
act randomly while with a lower one it will choose the action that with higher probability will bring the best reward. At the 
beginning of the agent´s learning it will be helpful to act more randomly (what is call Exploration) while at the final stages of the
training it will be better to Exploit the gained knowledge about the environment by the agent, so the best course of action is start
with a high value of epsilon (1) and gradually lowering it multipliying it by a decaying factor until it reaches a reasonable minimun.

Another technique used by the implemented algorithm is what is call Fixed Q targets. Its aim is avoid the moving Q-targets problem which
arises when using the same neural network to obtain the current prediction of the Q-value and the maximum possible value for the next 
state (target) and then calculate the loss between these two values. We are moving the prediction towards the target but, at the same 
time, we also move the target. This is why the agent initializes a second neural network (Q_target) with the same model and only updates
it with the weights of the Q network periodically after a number of fixed steps. This way the Q-target weights are fixed for most of the 
trainig.


### Chosen hyperparameters

* Learning rate (LR): `LR = 1e-3`

* Epsilon-greedy policy with decay: Epsilon is initialized to `1.0` and then starts decaying with factor of `0.98`. When it reaches the minimum of `0.005` it remains with this value for the rest of the training

* Discout Factor Gamma: The chosen value for the Discount Factor gamma is`0.99`

* `BATCH_SIZE = 64`

* `BUFFER_SIZE = 10.000`

* `TAU = 1e-3`


### Plot of Rewards

    ![Plot][image1]

### Ideas for Future Work

In order to improve the efficiency of the agent order algorithms can be tried. Those are:

* **Double DQN **
* **Prioritized experience replay**
* **Dueling DQN**


