import gym
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import tqdm


env = gym.make('MountainCar-v0')
eps = np.finfo(np.float32).eps.item()
seed= 43
np.random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

# env.reset()
# for _ in range(1000):
#     env.render()
#     observation,reward,done,info = env.step(env.action_space.sample())
#     print(observation,reward)
# env.close()

class Model(nn.Module):
    def __init__(self,observation_space,action_space,num_hidden):
        super(Model,self).__init__()

        self.common = nn.Linear(observation_space,num_hidden)
        self.actor = nn.Linear(num_hidden,action_space)
        self.critic = nn.Linear(num_hidden,1)

    def forward(self,observation):
        h = nn.ReLU(self.common(observation))

        action = self.actor(h)
        q = self.critic(h)

        return action,q



def run_episode(observation,model):

    action_prob = []
    values = []
    rewards = []
    done = False
    
    while not done:
        probabilities = 

def get_future_reward():
    pass

def compute_loss():
    pass

def train_step(initial_obs,model,optimizer):
    
    action_prob,values,rewards = run_episode(initial_obs,model)


def train_loop():
    model = Model(2,3,128)
    optimizer = torch.optim.Adam(lr=0.01)
    num_episodes=1000
    running_reward = 0
    gamma= 0.99

    with tqdm.trange(num_episodes) as t:
        for i in t:

            initial_obs = env.reset()
            episode_reward = train_step(initial_obs,model,optimizer)

            print(episode_reward)


train_loop()