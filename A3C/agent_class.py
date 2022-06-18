import numpy as np
import torch
import torch.nn.functional as F
from memory import Memory
from networks import ActorCritic
from torch.distributions import Categorical

class AgentProcess():
    def __init__(self, input_shape, n_actions, global_ac, 
                optimizer, gamma=0.99, tau=1.0):
        self.gamma = gamma
        self.tau = tau

        self.global_ac = global_ac # the shared global controller
        self.optimizer = optimizer

        self.memory = Memory()
        self.actor_critic = ActorCritic(input_shape, n_actions)

    def store_transition(self, reward, value, log_prob):
        self.memory.store_transition(reward, value, log_prob)
    
    def choose_action(self, obs):
        state = torch.tensor([obs], dtype=torch.float) # batchify
        probs, value = self.actor_critic(state)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # no need for .cpu() before .numpy() as we are on cpu already
        return action.numpy()[0], value, log_prob

    def learn(self, obs, done):
        # calculate rewards to go
        # then calculate GAE
        # then calculate losses
        # use backward(), then do optimizer step

        # load environment transitions that are used for gradient update
        rewards, values, log_probs = self.memory.sample_memory()

        self.optimizer.zero_grad()
        loss = self.calc_loss(obs, done, rewards, values, log_probs)
        loss.backward() # compute gradient of loss w.r.t. local agent's parameters
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 40) # in-place gradient norm clip
        self.copy_gradients_and_step() # take gradient step for global controller, and update local params
        self.memory.reset() # clear the memory after a gradient update

    # calculate the loss according to the A3C algorithm
    def calc_loss(self, new_state, done, rewards, values, log_probs):
        returns = self.calc_R(done, rewards, values)

        # if this transition is terminal, value is zero
        next_v = torch.zeros(1, 1) if done else self.actor_critic(torch.tensor([new_state], dtype=torch.float))[1]
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
            gae = delta_t[t] + (self.gamma*self.tau) * gae # TODO: why do we use gamma twice effectively?
            batch_gae[t] = gae
        batch_gae = torch.tensor(batch_gae, dtype=torch.float)

        # sum works better (gradient gets scaled by batch_size)
        actor_loss = -torch.sum(log_probs * batch_gae) # TODO: change to torch tensor sum operation (or isn't this just a dot product?)
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)
        entropy_loss = torch.sum(log_probs * torch.exp(log_probs)) # minimize negative entropy (maximizes entropy)

        total_loss = actor_loss + critic_loss + 0.01 * entropy_loss
        return total_loss

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

    def copy_gradients_and_step(self):
        for local_param, global_param in zip(self.actor_critic.parameters(), self.global_ac.parameters()):
            global_param.grad = local_param.grad
        self.optimizer.step() # update the central actor_critic with the gradients of the local model
        self.actor_critic.load_state_dict(self.global_ac.state_dict()) # load new global weights into local model