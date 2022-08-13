import torch
import torch.nn.functional as F
from networks import ActorValue

class Agent():
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, 
				 gamma=0.99):
		self.gamma = gamma
		self.lr = lr
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.actor_value = ActorValue(lr, input_dims, n_actions, fc1_dims, fc2_dims)

	def choose_action(self, state):
		probabilities, _ = self.actor_value(state)

		action_probs = torch.distributions.Categorical(probabilities)
		action = action_probs.sample()
		log_prob = action_probs.log_prob(action)

		return action.item(), log_prob

	def learn(self, state, reward, state_, log_prob, done):
		self.actor_value.optimizer.zero_grad()

		# convert reward to a torch tensor
		reward = torch.tensor(reward, dtype=torch.float).to(self.actor_value.device)

		# fetch critic's current Q-values for state and next state
		_, critic_value = self.actor_value(state)
		_, critic_value_ = self.actor_value(state_)
  
		# we want to nudge critic_value towards critic_target, not vice versa
		critic_value_ = critic_value_.detach()

		critic_target = reward + (1 - done) * self.gamma * critic_value_
		delta = critic_target - critic_value
		critic_loss = delta**2
		actor_loss = -log_prob*delta
		(actor_loss + critic_loss).backward()
		self.actor_value.optimizer.step()

	def save_models(self):
		self.actor_value.save_checkpoint()

	def load_models(self):
		self.actor_value.load_checkpoint()
