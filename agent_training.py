
from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from agent import Agent
import matplotlib.pyplot as plt

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


max_num_episodes = 15000

epsilon = 1.0
epsilon_min = 0.005
epsilon_decay = 0.98
gamma = 0.99
episode_scores = []

agent = Agent(gamma, action_size, state_size, epsilon, epsilon_min, epsilon_decay)

print("loop over episodes")

for episode in range(1, max_num_episodes+1):
    
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment 
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    
    for step in range(1000):
        
        action = agent.get_action(state)               # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        agent.agent_step(state, action, reward, next_state, done)
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
            
    episode_scores.append(score)
    
    if episode % 100 == 0:
        print('\rEpisode {}\tScore: {:.2f}'.format(episode, np.mean(episode_scores[episode - 100:])))
        
    if np.mean(episode_scores[episode - 100:]) >= 16:
        # Save weights
        torch.save(agent.Q.state_dict(), 'model_weights.pth')
        break
        
#Plot scores and save to image file
graph = plt.figure()
plt.plot([score for score in range(len(episode_scores))], episode_scores)
plt.ylabel('scores')
plt.xlabel('episodes')
plt.show()
graph.savefig('scores.jpg')        
        
#When finished, you can close the environment.
env.close()
        
