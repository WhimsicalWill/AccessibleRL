import torch
import torch.nn.functional as F
from networks import Actor, Value
from utils import ReplayBuffer

class Agent():
	def __init__(self, alpha, beta, input_shape, fc1_dims, n_actions, gamma=0.99):
		self.gamma = gamma

		self.actor = Actor(alpha, input_shape, n_actions, fc1_dims)
		self.value = Value(beta, input_shape, fc1_dims)
		self.memory = ReplayBuffer()

	def choose_action(self, state):
		state = torch.tensor([state], dtype=torch.float32).to(self.actor.device)
		probabilities = self.actor(state)
		dist = torch.distributions.Categorical(probabilities)
		action = dist.sample()

		return action.item()

	def store_transition(self, state, action, reward, done):
		self.memory.store_transition(state, action, reward, done)

	def calc_rewards_to_go(self, rewards, is_terminals, gamma):
		rewards_to_go = []
		discounted_reward = 0

		# accumulate rewards starting from the end of the buffer, working backwards
		for reward, done in zip(reversed(rewards), reversed(is_terminals)):
			if done: # reset to zero for terminal states
				discounted_reward = 0
			discounted_reward = reward + gamma * discounted_reward
			rewards_to_go.append(discounted_reward)
		return list(reversed(rewards_to_go)) # finally, reverse again to make forward-ordered

	def evaluate(self, states, actions):
		probs = self.actor(states)
		dist = torch.distributions.Categorical(probs)
		action_log_probs = dist.log_prob(actions).to(self.actor.device)
		state_values = self.value(states).to(self.actor.device)

		return action_log_probs, torch.squeeze(state_values)

	def learn(self):
		# convert reward to a torch tensor
		rewards_to_go = self.calc_rewards_to_go(self.memory.rewards, self.memory.is_terminals, self.gamma)
		rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(self.actor.device)
		rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / rewards_to_go.std()

		states = torch.tensor(self.memory.states, dtype=torch.float32).to(self.actor.device)
		actions = torch.tensor(self.memory.actions, dtype=torch.float32).to(self.actor.device)

		# Actor network update
		self.actor.optimizer.zero_grad()
		log_probs, state_values = self.evaluate(states, actions)
		advantages = rewards_to_go - state_values.detach() # detach for advantage computation
		actor_loss = -torch.mean(log_probs*advantages)
		actor_loss.backward()
		self.actor.optimizer.step()

		# Value network update
		self.value.optimizer.zero_grad()
		value_loss = F.mse_loss(state_values, rewards_to_go)
		value_loss.backward()
		self.value.optimizer.step()

		self.memory.clear()

	def save_models(self):
		self.actor.save_checkpoint()
		self.value.save_checkpoint()

	def load_models(self):
		self.actor.load_checkpoint()
		self.value.load_checkpoint()
