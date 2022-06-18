import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_dim=256, gamma=0.99, tau=1.0):
        super(ActorCritic, self).__init__()

        self.tau = tau
        self.gamma = gamma
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.pi = nn.Linear(hidden_dim, n_actions)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        pi = self.pi(x)
        v = self.v(x)

        # handle choosing action from pi
        probs = torch.softmax(pi, dim=1)

        return probs, v
    
    def calc_R(self, done, rewards, values):
        values = torch.cat(values).squeeze() # transform list to tensor

        # initialize the reward for calculating batch returns
        if len(values.size()) == 1: # batch of states
            R = values[-1] * (1 - int(done))
        elif len(values.size()) == 0:
            R = values * (1 - int(done))

        # iterate backwards over batch transitions
        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float).reshape(values.shape)

        return batch_return

    # calculate the loss according to the A3C algorithm
    def calc_loss(self, new_state, done, rewards, values, log_probs):
        returns = self.calc_R(done, rewards, values)

        # if this transition is terminal, value is zero
        next_v = torch.zeros(1, 1) if done else self.forward(torch.tensor([new_state], dtype=torch.float))[1]
        values.append(next_v.detach()) # detach from computation graph since it was just computed
        values = torch.cat(values).squeeze()
        log_probs = torch.cat(log_probs)
        rewards = torch.tensor(rewards)

        delta_t = rewards + self.gamma * values[1:] - values[:-1]
        n_steps = len(delta_t)
        batch_gae = np.zeros(n_steps) # initialize zero vector

        # calculate GAE using exponentially weighted deltas (O(n^2) time implementation)
        # for t in range(n_steps):
        #     for k in range(0, n_steps - t):
        #         temp = (self.gamma*self.tau)**k * delta_t[t+k]
        #         gae[t] += temp

        # O(n) time complexity implementation TODO: no need for gae variable
        gae = 0
        for t in reversed(list(range(n_steps))):
            gae = delta_t[t] + (self.gamma*self.tau) * gae
            batch_gae[t] = gae
        batch_gae = torch.tensor(batch_gae, dtype=torch.float)

        # sum works better (gradient gets scaled by batch_size)
        actor_loss = -torch.sum(log_probs * batch_gae) # TODO: change to torch tensor sum operation (or isn't this just a dot product?)
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)
        entropy_loss = torch.sum(log_probs * torch.exp(log_probs)) # minimize negative entropy (maximizes entropy)

        total_loss = actor_loss + critic_loss + 0.01 * entropy_loss
        return total_loss


