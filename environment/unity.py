import numpy as np
from environment.base import AbstractEnv
from unityagents import UnityEnvironment

ENVIRONMENTS = {
    'tennis': './Tennis_Windows_x86_64/Tennis.exe',
}

class UnityEnv(AbstractEnv):
    def __init__(self, env_name, train_mode):
        '''
            env_name : name of a unity environment to load
            env_mode : boolean, mode in which to use the environment 
        '''
        super().__init__()
        assert env_name, f'The Unity environment must be specified.  Possible names {ENVIRONMENTS.keys()}'
        assert env_name in ENVIRONMENTS, f'Unknown Unity environment \"{env_name}\".  Select from {ENVIRONMENTS.keys()}'
        
        print(f'Loading {env_name} in Train Mode {train_mode}')
        self.env = UnityEnvironment(file_name=ENVIRONMENTS[env_name])
        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.train_mode = train_mode
        states = self.reset()        
        self._state_dim = states.shape[1]
        self._num_agents = states.shape[0]
        self._action_dim = self.brain.vector_action_space_size

    def close(self):
        self.env.close()
        
    def reset(self):
        '''
            Reset the environment in the appropriate mode, returning the
            environment info, which has agents, and vector_observations,
            rewards, local_done
        '''
        return self.env.reset(train_mode=self.train_mode)[self.brain_name].vector_observations
        
    def step(self, actions):
        '''
            actions : vector of actions for each agent
            return next_states, rewards, dones
        '''
        # send actions for each agent to the environment
        env_info = self.env.step(actions)[self.brain_name]
        return env_info.vector_observations, np.array(env_info.rewards), np.array(env_info.local_done)

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def num_agents(self):
        return self._num_agents