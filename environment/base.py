from abc import ABC, abstractmethod, abstractproperty

class AbstractEnv(ABC):
    @abstractmethod
    def reset(self):
        '''
            Reset the environment to the initial state
        '''
        pass

    @abstractmethod
    def step(self, actions):
        '''
            actions: input actions 
            return 
                next_states : one or more next states for the environment(s)
                rewards : rewards for transition to next state(s)
                dones : whether or not the episode(s) finished after this transition
        '''
        pass

    @abstractproperty
    def state_dim(self):
        pass

    @abstractproperty
    def action_dim(self):
        pass

    @abstractproperty
    def num_agents(self):
        pass


