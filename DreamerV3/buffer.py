from collections import namedtuple
import torch
import numpy as np


transition = namedtuple('transition', (
    'state', 'action', 'reward', 'done', 'log_prob'
))

class Buffer:
    def __init__(self) -> None:
        self.reset()
    
    def store(self, state, action, reward, done, log_prob):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.log_prob.append(log_prob)
    
    def reset(self):
        self.storage = 
    
    def __len__(self):
        return len(self.state)
    
    def sample(self, batch_size, timesteps):
        idx = np.random.random_integers(0, len(self)-timesteps-1, batch_size)
