from utilities.network import *

class DDPGNetwork:
    '''
        Implementation of the deep deterministic policy gradient agent
        Adapted from From Udacity\deep-reinforcement-learning\ddpg-pendulum
        Which follows the paper https://arxiv.org/abs/1509.02971
        Also inspired by the modular implementation in https://github.com/ShangtongZhang/DeepRL
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.actor_local = config.actor_net_func()
        self.actor_target = config.actor_net_func()
        self.critic_local = config.critic_net_func()
        self.critic_target = config.critic_net_func()
        # transfer all the networks to the device
        self.actor_local.to(config.device)
        self.actor_target.to(config.device)
        self.critic_local.to(config.device)
        self.critic_target.to(config.device)
        self.actor_opt = config.actor_opt_func(self.actor_local.parameters())
        self.critic_opt = config.critic_opt_func(self.critic_local.parameters())

    def learn(self, experiences, gamma):
        '''
            Perform one mini-batch update of the networks
            Improve : batch normalization of the input state could be used as 
            described in the paper
        '''
        config = self.config
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), config.gradient_clip)
        self.critic_opt.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_local.parameters(), config.gradient_clip)
        self.actor_opt.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, config.tau)
        self.soft_update(self.actor_local, self.actor_target, config.tau)  

    def act(self, state):
        '''
            Returns actions for given state as per current policy.
            Improve : random noise could be added to the actions
        '''      
        state = self.config.to_tensor(state)
        self.actor_local .eval()
        # get a new action from the actor
        with torch.no_grad():
            action = to_np(self.actor_local(state))
        self.actor_local.train()

        # clip to -1,1 although the tanh on the actor output should have done this
        return np.clip(action, -1, 1)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Moves the target slowly towards the local network which is being trained

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



