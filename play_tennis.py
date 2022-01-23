import argparse
import os
import json
import torch
import torch.nn as nn
from utilities.config import Config
from environment.unity import UnityEnv
from network.bodies import Actor, FCBody, Critic
from network.ddpg import DDPGNetwork
from agent.memory_replay import MemoryReplayAgent

parser = argparse.ArgumentParser(description="Train an agent to solve the tennis environment")
parser.add_argument('config_file', metavar='f', help="Path to a json configuration file")
parser.add_argument('--network_file', metavar='n', help="Path to a network file", required=False, default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        raise Exception('File {} does not exist'.format(args.config_file))

    network_file = args.network_file
    if network_file and not os.path.exists(network_file):
        raise Exception('File {} does not exist'.format(args.network_file)) 

    with open(args.config_file) as f:
        args = json.load(f)
        config = Config()
        config.merge(args)

        config.env = UnityEnv(config.env_name, config.train_mode)

        config.actor_net_func = lambda: Actor(
            FCBody(config.env.state_dim, gate_func=nn.ReLU), 
            config.env.action_dim,
            nn.Tanh)

        config.critic_net_func = lambda: Critic(
            config.env.state_dim,
            config.env.action_dim)

        config.actor_opt_func = lambda params: \
            torch.optim.Adam(params, config.actor_learning_rate)

        config.critic_opt_func = lambda params: \
            torch.optim.Adam(params, config.critic_learning_rate, weight_decay=config.weight_decay)
        
        config.network_func = lambda: DDPGNetwork(config)

        agent = MemoryReplayAgent(config)
        if network_file:
            print('Loading network file {}'.format(network_file))
            agent.load(network_file)

        if config.train_mode:
            agent.train()
        else:
            agent.run()


        