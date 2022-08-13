import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Critic(nn.Module):
	def __init__(self, beta, input_dims, n_actions, fc1_dims=256, chkpt_dir='tmp/dqn'):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(*input_dims, fc1_dims)
		self.value = nn.Linear(fc1_dims, n_actions)

		self.chkpt_file = f"{chkpt_dir}/dqn_critic"

		self.optimizer = optim.Adam(self.parameters(), lr=beta)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state): 	 
		x = F.relu(self.fc1(state))
		action_values = self.value(x)
		return action_values

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.chkpt_file)

	def load_checkpoint(self):
		self.load_state_dict(torch.load(self.chkpt_file))
