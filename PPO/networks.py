import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Value(nn.Module):
	def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, chkpt_dir='tmp/ppo'):
		super(Value, self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		self.chkpt_file = f"{chkpt_dir}/critic_ppo"

		self.fc1 = nn.Linear(*input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)

		self.bn1 = nn.LayerNorm(fc1_dims)
		self.bn2 = nn.LayerNorm(fc2_dims)

		self.v = nn.Linear(fc2_dims, 1)
		self.softmax = nn.Softmax(dim=-1)

		self.optimizer = optim.Adam(self.parameters(), lr=beta)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		x = self.fc1(state)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = self.bn2(x)
		x = F.relu(x)
		state_value = self.v(x)
		return state_value

	def save_checkpoint(self):
		print(' ... saving checkpoint ...')
		torch.save(self.state_dict(), self.chkpt_file)

	def load_checkpoint(self):
		print(' ... loading checkpoint ...')
		self.load_state_dict(torch.load(self.chkpt_file))
  
class Actor(nn.Module):
	def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, chkpt_dir='tmp/ppo'):
		super(Actor, self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions # this really should be 'action_dim' since it's not nec. discrete
		self.chkpt_file = f"{chkpt_dir}/actor_ppo"

		self.fc1 = nn.Linear(*input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)

		self.bn1 = nn.LayerNorm(fc1_dims)
		self.bn2 = nn.LayerNorm(fc2_dims)

		self.pi = nn.Linear(fc2_dims, n_actions)
		self.softmax = nn.Softmax(dim=-1)

		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, state):
		x = self.fc1(state)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = self.bn2(x)
		x = F.relu(x)
		probs = self.softmax(self.pi(x))
		return probs

	def save_checkpoint(self):
		print(' ... saving checkpoint ...')
		torch.save(self.state_dict(), self.chkpt_file)

	def load_checkpoint(self):
		print(' ... loading checkpoint ...')
		self.load_state_dict(torch.load(self.chkpt_file))