import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
# import torch
# import torch.nn.functional as F
# import numpy as np
# from ou_noise import OUActionNoise
# from buffer import ReplayBuffer
# from networks import ActorNetwork, CriticNetwork

# class Agent():
# 	def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
# 					max_size=1000000, fc1_dims=400, fc2_dims=300,  batch_size=64):
# 		self.alpha = alpha
# 		self.beta = beta
# 		self.gamma = gamma
# 		self.tau = tau
# 		self.batch_size = batch_size

# 		self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions, "actor")
# 		self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, "critic")
# 		self.memory = ReplayBuffer(max_size, input_dims, n_actions)

# 	def store_transition(self, state, action, reward, state_, done):
# 		self.memory.store_transition(state, action, reward, state_, done)

# 	def choose_action(self, state):
# 		# use current policy to sample action
# 		state = torch.FloatTensor(state).to(device)
# 		actor_output = self.actor(state)
# 		dist = Categorical(actor_output)
# 		action = dist.sample()

# 		self.store_transition(state, action.detach(), dist.log_prob(action).detach())

# 		# these data must be added as tensors
# 		self.memory.states.append(state)
# 		self.memory.actions.append(action.detach())
# 		self.memory.log_probs.append(dist.log_prob(action).detach())
# 		return action.detach().item()

# 	def learn(self):
# 		if self.memory.mem_ctr < self.batch_size:
# 			return # don't learn until we can sample at least a full batch

# 		# Sample memory buffer uniformly
# 		states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

# 		# Convert from numpy arrays to torch tensors for computation graph
# 		states = torch.tensor([states], dtype=torch.float).to(self.actor.device)
# 		actions = torch.tensor([actions], dtype=torch.float).to(self.actor.device)
# 		states_ = torch.tensor([states_], dtype=torch.float).to(self.actor.device)
# 		rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
# 		done = torch.tensor(done, dtype=torch.float).to(self.actor.device)
		
# 		# Update the critic to minimize the MSE w.r.t. the target critic
# 		self.critic.optimizer.zero_grad()
# 		target_actions = self.target_actor(states_)
# 		target_q = rewards + (1.0 - done) * (self.gamma * self.target_critic(states_, target_actions).view(-1))
# 		target_q = target_q.view(self.batch_size, 1) # shape (B, 1)
# 		critic_loss = F.mse_loss(self.critic(states, actions), target_q)
# 		critic_loss.backward()
# 		self.critic.optimizer.step()

# 		# Update the actor greedily w.r.t to the target critic
# 		self.actor.optimizer.zero_grad()
# 		actor_output = self.actor(states)
# 		actor_loss = -torch.mean(self.target_critic(states, actor_output))
# 		actor_loss.backward()
# 		self.actor.optimizer.step()

# 		# Do a soft update after a learning step
# 		self.update_agent_parameters()

class Agent:
	def __init__(self, alpha, beta, input_shape, n_actions, batch_size=64, gamma=0.99, eps=0.2, fc1_dims=256):
		self.input_shape = input_shape
		self.n_actions = n_actions
		self.batch_size = batch_size
		self.gamma = 0.99
		self.eps = eps
		self.fc1_dims = fc1_dims

		self.critic = Critic(beta, input_shape, fc1_dims, fc2_dims, n_actions).to(device)
		self.actor = Actor(alpha, input_shape, fc1_dims, fc2_dims, n_actions).to(device)
		self.memory = ReplayBuffer(max_size, input_dims, n_actions)

		# lr_critic = 0.001       # learning rate for critic network
		# lr_actor = 0.0003       # learning rate for actor network

	def store_transition(self, state, action, reward, state_, done):
		self.memory.store_transition(state, action, reward, state_, done)

	def choose_action(self, state):
		state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
		probs = self.actor(state)
		dist = Categorical(probs)
		action = dist.sample()
  
		# TODO: return information and exec store_transition in main loop
		# TODO: track log_prob

	def learn(self):
		rewards = []
		discounted_reward = 0
		for reward, done in zip(reversed(self.data.rewards), reversed(self.data.is_terminals)):
			if done:
				discounted_reward = 0
			discounted_reward = reward + self.gamma * discounted_reward
			# print("reward", reward, " disc", discounted_reward)
			rewards.insert(0, discounted_reward)

		rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

		rewards = (rewards - rewards.mean()) / rewards.std()

		# convert list to tensor
		old_states = torch.squeeze(torch.stack(self.data.states, dim=0)).detach()
		old_actions = torch.squeeze(torch.stack(self.data.actions, dim=0)).detach()
		old_log_probs = torch.squeeze(torch.stack(self.data.log_probs, dim=0)).detach().to(device)

		# evaluate a new policy at the old transitions and take a opt step using gradient of loss
		for i in range(self.num_epochs):
			log_probs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

			ratios = torch.exp(log_probs - old_log_probs)

			# advantages
			state_values = torch.squeeze(state_values).detach()
			advantages = rewards - state_values

			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1-self.eps, 1+self.eps) * advantages

			loss = -1 * torch.min(surr1, surr2) + 0.05 * self.MseLoss(rewards, state_values) - 0.01 * dist_entropy
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()
			self.data.clear()