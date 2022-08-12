import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
		log_prob = dist.log_prob(action).detach()
  
		return action.detach().item(), log_prob

	def calc_rewards_to_go(reward, is_terminals, gamma):
		rewards = []
		discounted_reward = 0
		for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
			if done:
				discounted_reward = 0
			discounted_reward = reward + self.gamma * discounted_reward
			# print("reward", reward, " disc", discounted_reward)
			rewards.insert(0, discounted_reward)

	def learn(self, pi_update_iter, value_update_iter):
		# calculate rewards to go
		rewards_to_go = calc_rewards_to_go(self.memory.reward, self.memory.is_terminals, self.gamma)

		rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
		rewards = (rewards - rewards.mean()) / rewards.std()

		# convert list to tensor
		old_states = torch.squeeze(torch.stack(self.memory.states, dim=0)).detach()
		old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach()
		old_log_probs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(device)

		# evaluate a new policy at the old transitions and take a opt step using gradient of loss
		for i in range(self.pi_update_iter):
			self.actor.optimizer.zero_grad()
			log_probs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

			ratios = torch.exp(log_probs - old_log_probs)

			# advantages
			state_values = torch.squeeze(state_values).detach()
			advantages = rewards - state_values

			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1-self.eps, 1+self.eps) * advantages

			loss = -1 * torch.min(surr1, surr2) + 0.05 * self.MseLoss(rewards, state_values) - 0.01 * dist_entropy
			loss.mean().backward()
			self.optimizer.step()
			self.memory.clear()