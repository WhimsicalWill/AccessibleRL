import torch
import torch.nn as nn

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_size):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(state_dim, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, action_dim)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, state):
		x = self.fc1(state)
		x = self.fc2(F.relu(x))
		x = self.fc3(F.relu(x))
		# softmax output to ensure sum of probabilities is one
		return self.softmax(x)

class Critic(nn.Module):
	def __init__(self, state_dim, hidden_size):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(state_dim, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, 1)

	def forward(self, state):
		x = self.fc1(state)
		x = self.fc2(F.relu(x))
		x = self.fc3(F.relu(x))
		return x