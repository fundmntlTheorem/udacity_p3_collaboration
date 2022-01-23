
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Udacity Collaboration and Competition Project 3

### Introduction

The goal of this project is to train a pair of agents to play tennis.  The agents must bounce a ball back and forth over a net.  Hitting the ball over the net gives the agent a reward of +0.1.  If the ball hits the ground or goes out of bounds, the agent receives a reward of -0.01.  The longer the agents keep the ball in play, the higher the reward.  The environment is considered to be solved when the average maximum score over both agents is > +0.5 over 100 episodes.

![Trained Agent][image1]

### Environment Details

The environment is provided by [Unity](https://unity.com/), a company that specializes in building worlds that can be used for video game development, simulation, animation, and architecture/design.  The following is the description of the state space and actions available to the agent:

*The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.*

### Dependencies

1. Download the x64 windows environment for the single agent, from [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip).

2. The code expects the Tennis.exe file to be located in the following directory of the repo  "./Tennis_Windows_x86_64/Tennis.exe"

3. Create a new conda environment with the provided requirements.txt file. Ex. conda create --name <env> --file requirements.txt

## Using the Code

The code may be run using the command `python play_tennis.py <config.json file> [network_file.pth]`.  The network file is an optional parameter that will first load a previous file. 

### Train/Run Mode

Example `python play_tennis.py config.json`

The arguments to the program are provided using a .json file.  See the utilities/config.py file for the default parameters.  Setting `train_mode: true` will train the agents.  Setting `train_mode: false` will simply run the agents for a single episode without training the networks.

```python
{
    "env_name": "tennis",
    "train_mode": true,
    "device": "cuda",
    "actor_learning_rate": 1e-4,
    "critic_learning_rate": 1e-4,
    "batch_size": 32,
    "gamma": 0.99,
    "tau": 1e-3,
    "num_episodes": 5000
}
```