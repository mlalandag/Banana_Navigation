
from unityagents import UnityEnvironment
import numpy as np
import random
import torch
import time
from agent import Agent

# please do not modify the line below
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)


max_num_episodes = 25

epsilon = 0.005
epsilon_min = 0.005
epsilon_decay = 1
gamma = 1

agent = Agent(gamma, action_size, state_size, epsilon, epsilon_min, epsilon_decay)

agent.Q.load_state_dict(torch.load('model_weights.pth'))

for episode in range(1, max_num_episodes+1):
    
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment 
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    
    while True:
        
        action = agent.get_action(state)               # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        time.sleep(0.05)
        if done:                                       # exit loop if episode finished
            break
    
    print('\rEpisode {}\tScore: {:.2f}'.format(episode, score))
     
        
#When finished, you can close the environment.
env.close()
        
