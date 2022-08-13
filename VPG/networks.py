import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
	def __init__(self, alpha, input_dims, n_actions, fc1_dims=256, chkpt_dir='tmp/vpg'):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(*input_dims, fc1_dims)
		self.pi = nn.Linear(fc1_dims, n_actions)
  
		self.chkpt_file = f"{chkpt_dir}/vpg_actor"
  
		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state): 	 
		x = F.relu(self.fc1(state))
		action_probs = F.softmax(self.pi(x))
		return action_probs

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.chkpt_file)

	def load_checkpoint(self):
		self.load_state_dict(torch.load(self.chkpt_file))

class Value(nn.Module):
	def __init__(self, beta, input_dims, fc1_dims=256, chkpt_dir='tmp/vpg'):
		super(Value, self).__init__()
		self.fc1 = nn.Linear(*input_dims, fc1_dims)
		self.value = nn.Linear(fc1_dims, 1)

		self.chkpt_file = f"{chkpt_dir}/vpg_value"

		self.optimizer = optim.Adam(self.parameters(), lr=beta)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state): 	 
		x = F.relu(self.fc1(state))
		state_value = F.softmax(self.value(x))
		return state_value

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.chkpt_file)

	def load_checkpoint(self):
		self.load_state_dict(torch.load(self.chkpt_file))