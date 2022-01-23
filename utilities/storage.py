from collections import deque, namedtuple
import torch
import numpy as np
import random

class Storage:
    '''Storage buffer that can be used for PPO type roll-outs'''
    def __init__(self, memory_size, keys):
        self.memory_size = memory_size
        self.keys = keys
        self.memory = {k: deque(maxlen=memory_size+1) for k in keys}

    def feed(self, data_dict):
        '''
            Add new information for the items in data_dict
            data_dict is a dictionary of key: scalar values
        '''
        for k,v in data_dict.items():
            self.memory[k].append(v)
            
    def operate(self, keys, operator):
        '''
            apply an operation to the data some key in the memory
        '''
        for k in keys:
            self.memory[k] = operator(self.memory[k])
    
    def __getitem__(self, key):
        return self.memory[key]
    
    def get_extractor(self, keys, device):
        '''
            Convert the deques to torch stacks
        '''     
        # make convenient tuple for indexing the items
        experience = namedtuple('Experience', keys)
        # do the conversion to stacks and transfer to device just one time
        stacks = []
        for k in keys:
            # convert the deque to list, then to a stack and transfer to the device
            # it's very important to detach and create a new tensor, otherwise
            # when the backward call happens, it erases
            stacks.append(torch.stack(list(self.memory[k])).to(device).detach())
                
        def extract(indices):
            '''
                Extract the items at the indices and return
                the named tuple, indexed by the keys
            '''            
            batch = []
            for items in stacks:
                batch.append(torch.index_select(items, dim=0, index=indices))
            return experience(*batch)
        return extract

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        device = self.device
        # seed set in config.py
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)