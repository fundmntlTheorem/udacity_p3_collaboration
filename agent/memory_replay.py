import numpy as np
import torch
import torch.nn as nn
from .base import BaseAgent
from utilities.storage import ReplayBuffer

class MemoryReplayAgent(BaseAgent):
    '''
        This object is basically a shell that can train
        for an agent that can utilize a experience memory buffer
    '''
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.env = config.env
        self.network = config.network_func()
        self.reset()
        self.memory = ReplayBuffer(self.env.action_dim, config.buffer_size, config.batch_size, config.device)

    def reset(self):
        self.state = self.env.reset()

    def step(self, skip_training=False):
        '''
            Save experience in replay memory, and use random sample from buffer to learn.
            return the reward received on this step
        '''
        # perform an action for each agent
        action = [self.network.act(state) for state in self.state]
        next_state, reward, done = self.env.step(action)

        if not skip_training:
            # Save the experience to the memory buffer, for each agent
            for s,a,r,n,d in zip(self.state, action, reward, next_state, done):
                self.memory.add(s, a, r, n, d)

            # Learn, if enough samples are available in memory
            if len(self.memory) > self.config.batch_size:
                experiences = self.memory.sample()
                self.network.learn(experiences, self.config.gamma)

        # store the next state
        self.state = next_state

        return reward, done

    def save(self, file_name, metrics):
        '''
            Save the networks to a file
        '''
        torch.save(
            {
                'actor': self.network.actor_local.state_dict(),
                'critic': self.network.critic_local.state_dict(),
                'metrics': metrics,
            },
            file_name)

    def load(self, file_name):
        '''
            Restore the actor/critic networks
        '''
        info = torch.load(file_name)
        self.actor_local.load_state_dict(info['actor'])
        self.critic_local.load_state_dict(info['critic'])

    

    