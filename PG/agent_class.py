import torch
import torch.nn.functional as F
from networks import ActorCritic

class Agent():
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, 
				 gamma=0.99):
		self.gamma = gamma
		self.lr = lr
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.actor_critic = ActorCritic(lr, input_dims, n_actions, fc1_dims, fc2_dims)

	def choose_action(self, state):
		probabilities, _ = self.actor_critic(state)

		action_probs = torch.distributions.Categorical(probabilities)
		action = action_probs.sample()
		log_prob = action_probs.log_prob(action)

		return action.item(), log_prob

	def learn(self, state, reward, state_, log_prob, done):
		self.actor_critic.optimizer.zero_grad()

		# convert reward to a torch tensor
		reward = torch.tensor(reward, dtype=torch.float).to(self.actor_critic.device)

		# fetch critic's current Q-values for state and next state
		_, critic_value = self.actor_critic(state)
		_, critic_value_ = self.actor_critic(state_)

		critic_target = reward + (1 - done) * self.gamma * critic_value_
		delta = critic_target - critic_value
		critic_loss = delta**2
		actor_loss = -log_prob*delta
		(actor_loss + critic_loss).backward()
		self.actor_critic.optimizer.step()

	def save_models(self):
		self.actor_critic.save_checkpoint()
