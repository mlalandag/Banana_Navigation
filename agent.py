import torch
import time
import random
import numpy as np
from replay_buffer import ReplayBuffer
from model import SNN

class Agent:
    
    def __init__(self, gamma, action_size, state_size, epsilon, epsilon_min, epsilon_decay):
        
        print("Agent init")
        
        self.action_size = action_size
        print('Number of actions:', self.action_size)
        self.state_size  = state_size
        print('Number of states:', self.state_size)
        
        # Q-Network
        print("Set Neural Networks")
        self.Q = SNN(self.state_size, self.action_size)
        self.Q_target = SNN(self.state_size, self.action_size)
        print("Set optimizer")
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.step = 0
        self.tau = 1e-3
        
        self.memory = ReplayBuffer(self.action_size, 10000, 64, 0)
        
    def get_action(self, state):
        
        state = torch.from_numpy(state).float().unsqueeze(0)

        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        
        self.Q.eval()
        with torch.no_grad():
            action_values = self.Q(state).data.numpy()
        self.Q.train()
        
        if np.random.random() > self.epsilon:
            return np.argmax(action_values).astype(int)
        else:
            return np.random.choice([a for a in range(self.action_size)])
                
        
    def learn(self, experience, gamma):
        
        states, actions, rewards, next_states, dones = experience
        
        td_target = self.Q_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + gamma * td_target * (1 - dones)
        Q_expected = self.Q(states).gather(1, actions)
        td_error  = torch.nn.functional.mse_loss(Q_expected, Q_targets)
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()
        
        for target, local in zip(self.Q_target.parameters(), self.Q.parameters()):
            target.data.copy_(self.tau * local.data + (1 - self.tau) * target.data)
        
    def agent_step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.step = (self.step + 1)%64
        if self.step == 0:
            if len(self.memory) > 64:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)