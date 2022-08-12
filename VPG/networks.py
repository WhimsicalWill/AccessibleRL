import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorValue(nn.Module):
	def __init__(self, lr, input_dims, n_actions, name, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/vpg'):
		super(ActorCritic, self).__init__()
		self.fc1 = nn.Linear(*input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.pi = nn.Linear(fc2_dims, n_actions)
		self.v = nn.Linear(fc2_dims, 1)
  
		self.chkpt_file = f"{chkpt_dir}/vpg_chkpt"
  
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state): 	 
		state = torch.tensor([state], dtype=torch.float).to(self.device)
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		action_probs = F.softmax(self.pi(x))
		state_value = self.v(x)
	
		return action_probs, state_value

	def save_checkpoint(self):
		torch.save(self.state_dict(), self.chkpt_file)

	def load_checkpoint(self):
		self.load_state_dict(torch.load(self.chkpt_file))
