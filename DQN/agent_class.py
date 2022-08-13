import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from networks import Critic
from utils import ReplayBuffer

class Agent:
	def __init__(self, beta, input_shape, n_actions, batch_size=64, gamma=0.99,
				eps=.01, fc1_dims=256, max_size=10000): 
		self.input_shape = input_shape
		self.n_actions = n_actions
		self.batch_size = batch_size
		self.gamma = gamma
		self.eps = eps

		self.critic = Critic(beta, input_shape, n_actions, fc1_dims)
		self.memory = ReplayBuffer(max_size, input_shape)

	def store_transition(self, state, action, reward, state_, done):
		self.memory.store_transition(state, action, reward, state_, done)

	def choose_action(self, state):
		if torch.rand(1).item() < self.eps:
			print("RANDOM ACTION")
			action = torch.randint(low=0, high=self.n_actions, size=(1,))
		else:
			state = torch.tensor([state], dtype=torch.float).to(self.critic.device)
			action_values = self.critic(state)
			action = torch.argmax(action_values)

		return action.detach().item()

	def learn(self):
		if self.memory.mem_ctr < self.batch_size:
			return

		states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

		states = torch.tensor(states, dtype=torch.float32).to(self.critic.device)
		states_ = torch.tensor(states_, dtype=torch.float32).to(self.critic.device)
		rewards = torch.tensor(rewards, dtype=torch.float32).to(self.critic.device)
		done = torch.tensor(done, dtype=torch.float32).to(self.critic.device)
		batch_index = np.arange(self.batch_size, dtype=np.int32)

		# Critic update using max of next-step Q-values and MSE loss
		self.critic.optimizer.zero_grad()
		critic_value = self.critic(states)[batch_index, actions]
		critic_target = rewards + (1 - done) * self.gamma * torch.max(self.critic(states), dim=-1)[0]
		loss = F.mse_loss(critic_value, critic_target)
		loss.backward()
		self.critic.optimizer.step()

	def save_models(self):
		self.critic.save_checkpoint()

	def load_models(self):
		self.critic.load_checkpoint()

