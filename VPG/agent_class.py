import torch
import torch.nn.functional as F
from networks import ActorValue
from utils import ReplayBuffer

class Agent():
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, 
				 gamma=0.99):
		self.gamma = gamma
		self.lr = lr
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims

		self.actor_value = ActorValue(lr, input_dims, n_actions, fc1_dims, fc2_dims)
		self.memory = ReplayBuffer()

	def choose_action(self, state):
		state = torch.tensor(state, dtype=torch.float32).to(self.actor_value.device)
		probabilities, _ = self.actor_value(state)

		action_probs = torch.distributions.Categorical(probabilities)
		action = action_probs.sample()
		log_prob = action_probs.log_prob(action)

		return action.item(), log_prob

	def store_transition(self, state, reward, log_prob, done):
		self.memory.store_transition(state, reward, log_prob, done)

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

	def learn(self):
		print("Learning update")

		self.actor_value.optimizer.zero_grad()

		# convert reward to a torch tensor
		rewards_to_go = self.calc_rewards_to_go(self.memory.rewards, self.memory.is_terminals, self.gamma)
		rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(self.actor_value.device)
		rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / rewards_to_go.std()

		states = torch.tensor(self.memory.states, dtype=torch.float32).to(self.actor_value.device)
		log_probs = torch.tensor(self.memory.log_probs, dtype=torch.float32).to(self.actor_value.device)

		# print(f"Shapes: {rewards_to_go.shape} | {states.shape} | {log_probs.shape} | {len(self.memory.is_terminals)}")

		# TODO: normalize rewards and put on gpu (?)

		_, state_values = self.actor_value(states)
		state_values = torch.squeeze(state_values)
		advantages = rewards_to_go - state_values.detach() # detach for advantage computation

		actor_loss = -torch.mean(log_probs*advantages)
		value_loss = F.mse_loss(state_values, rewards_to_go)
		print(f"Losses: {actor_loss} | {value_loss}")
		(actor_loss + value_loss).backward()
		self.actor_value.optimizer.step()

	def save_models(self):
		self.actor_value.save_checkpoint()

	def load_models(self):
		self.actor_value.load_checkpoint()
