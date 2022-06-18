import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                    chkpt_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = f"{chkpt_dir}/{name}_td3"

        self.fc1 = nn.Linear(input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device) # put the model on the device

    def forward(self, state, action):
        q1_action_value = self.fc1(torch.cat([state, action], dim=1)) # concat state and action for all batches
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1_action_value = self.q(q1_action_value)
        return q1_action_value

    def save_checkpoint(self):
        print(' ... saving checkpoint ...')
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print(' ... loading checkpoint ...')
        self.load_state_dict(torch.load(self.chkpt_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = f"{self.chkpt_dir}/{self.name}_best"
        torch.save(self.state_dict(), checkpoint_file)

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions,
                    name, chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions # this really should be 'action_dim' since it's not nec. discrete
        self.name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = f"{chkpt_dir}/{name}_td3"

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        prob = torch.tanh(self.mu(prob)) # maps fc2dims to action_dims then restricts to interval [-1, 1]
        return prob

    def save_checkpoint(self):
        print(" ... saving checkpoint ...")
        torch.save(self.state_dict(), self.chkpt_file)
    
    def load_checkpoint(self):
        print(" ... loading checkpoint ...")
        self.load_state_dict(torch.load(self.chkpt_file))
    
    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = f"{self.chkpt_dir}/{self.name}_best"
        torch.save(self.state_dict(), checkpoint_file)