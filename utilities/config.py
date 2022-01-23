import torch
import random
from .network import to_tensor

'''
    Taken from https://github.com/ShangtongZhang/DeepRL
'''

class Config:
    def __init__(self):
        self.env_name = None
        self.batch_size = 8
        self.buffer_size = int(1e5)
        self.gamma = 0.99
        self.tau = 1e-3
        self.learning_rate = 1e-3
        self.actor_learning_rate = 1e-4
        self.critic_learning_rate = 1e-4
        self.weight_decay = 0.0
        self.train_mode = True
        self.env = None
        self.network = None
        self.device = 'cpu'
        self.to_tensor = None        
        self.actor_opt = None
        self.critic_opt = None
        self.actor_net_func = None
        self.critic_net_func = None
        self.seed = None    
        self.gradient_clip = 5
        self.reward_window_size = 100
        self.max_steps = 1000
        self.num_episodes = 1000
        self.log_interval = 100
        self.save_interval = 100

    def validate_device(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Device {self.device}')
        self.to_tensor = lambda x: to_tensor(x, self.device)

    def merge(self, config_dict):
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
        self.validate_device()
        # initialize a generator with either the passed seed, or none,
        # which makes a fresh, unpredictable seed
        random.seed(self.seed)