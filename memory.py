import numpy as np
import torch
import random
from collections import deque
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'next_state_unallowed'))


class Memory:
    
    """
    Class for the uniform experience replay memory.
    """
    
    def __init__(self, max_size):
        """
        Description
        -------------
        Constructor of class Memory.
        
        Parameters & Attributes
        -------------
        max_size   : Int, the maximum size of the replay memory.
        buffer     : collections.deque object of maximum length max_size, the container representing the replay memory.
        """
        
        self.buffer = deque(maxlen = max_size)
    
    def add(self, state, action, reward, next_state):
        """
        Description
        -------------
        Add experience to the replay buffer.
        
        Parameters
        -------------
        experience : 4-tuple representing a transition (state, action, reward, next_state).
                     - state      : Object of class State representing the state.
                     - action     : Int in {0, ..., d-1, d, d+1}, the action.
                     - reward     : Float, the reward.
                     - next_state : Object of class State representing the next state.
                     
        Returns
        -------------
        """
        
        next_state_unallowed = np.zeros(len(next_state.categories) + 2, dtype=np.float32)
        next_state_unallowed[next_state.observed] = 1
        self.buffer.append((torch.from_numpy(state.values_encoded), [action], torch.tensor([reward], dtype=torch.float32), 
                            torch.from_numpy(next_state.values_encoded), torch.from_numpy(next_state_unallowed).view(1, -1)))
    
    def sample(self, batch_size):
        """
        Description
        -------------
        Randomly sample "batch_size" experiences from the replay buffer.
        
        Parameters
        -------------
        batch_size : Int, the number of experiences to sample.
        
        Returns
        -------------
        Named tuple representing the sampled batch.
        """
        
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))






