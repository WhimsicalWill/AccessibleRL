import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Agent:
	def __init__(self, alpha, beta, input_shape, n_actions, batch_size=64, gamma=0.99, 
				eps=0.2, fc1_dims=256, fc2_dims=256):
		self.input_shape = input_shape
		self.n_actions = n_actions
		self.batch_size = batch_size
		self.gamma = 0.99
		self.eps = eps
		self.fc1_dims = fc1_dims

		self.critic = Critic(beta, input_shape, fc1_dims, fc2_dims, n_actions).to(device)
		self.actor = Actor(alpha, input_shape, fc1_dims, fc2_dims, n_actions).to(device)
		self.memory = ReplayBuffer(max_size, input_dims, n_actions)

	def store_transition(self, state, action, reward, state_, done):
		self.memory.store_transition(state, action, reward, state_, done)

	def choose_action(self, state):
		state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
		probs = self.actor(state)
		dist = Categorical(probs)
		action = dist.sample()
		log_prob = dist.log_prob(action).detach()
  
		return action.detach().item(), log_prob

	def calc_rewards_to_go(rewards, is_terminals, gamma):
		rewards_to_go = []
		discounted_reward = 0

		# accumulate rewards starting from the end of the buffer, working backwards
		for reward, done in zip(reversed(rewards), reversed(is_terminals)):
			if done: # reset to zero for terminal states
				discounted_reward = 0
			discounted_reward = reward + gamma * discounted_reward
			rewards_to_go.append(discounted_reward)
		return reversed(rewards_to_go) # finally, reverse again to make forward-ordered

	def evaluate(self, states, actions):
		state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
		probs = self.actor(state)
		dist = Categorical(probs)
		action_log_probs = dist.log_prob(action).to(self.actor.device)
		state_values = self.critic(states).to(self.actor.device)
		return action_log_probs, state_values, dist.entropy()

	def learn(self, pi_update_iter, value_update_iter):
		# calculate rewards to go
		rewards_to_go = calc_rewards_to_go(self.memory.reward, self.memory.is_terminals, self.gamma)

		# convert to torch tensor and normalize
		rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(device)
		rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / rewards_to_go.std()

		# convert lists to tensors and add a dimension to get (B, 1) shape
		# TODO: debug tensor shapes
		old_states = torch.tensor([self.memory.states], dtype=torch.float32).to(device)
		old_actions = torch.tensor([self.memory.actions], dtype=torch.float32).to(device)
		old_log_probs = torch.tensor([self.memory.log_probs], dtype=torch.float32).to(device)

		# evaluate a new policy at the old transitions
		for _ in range(self.pi_update_iter):
			self.actor.optimizer.zero_grad()
			log_probs, state_values, _ = self.evaluate(old_states, old_actions)
			ratios = torch.exp(log_probs - old_log_probs)
			clipped_ratios = torch.clamp(ratios, 1-self.eps, 1+self.eps)

			# calculate advantage function
			state_values = torch.squeeze(state_values).detach()
			advantages = rewards_to_go - state_values

			surr1 = ratios * advantages
			surr2 = clipped_ratios * advantages

			actor_loss = -torch.min(surr1, surr2).mean()
			actor_loss.backward()
			self.optimizer.step()

		for _ in range(self.value_update_iter):
			self.critic.optimizer.zero_grad()
			_, state_values, _ = self.evaluate(old_states, old_actions)

			value_loss = F.mse_loss(state_values, rewards_to_go)
			value_loss.backward()
			self.critic.optimizer.step() 
		
		self.memory.clear() # clear replay buffer after learning update

	def save_models(self):
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()

	def load_models(self):
		self.actor.load_checkpoint()
		self.critic.load_checkpoint()
