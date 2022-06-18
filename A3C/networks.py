import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_dim=256):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.pi = nn.Linear(hidden_dim, n_actions)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        pi = self.pi(x)
        v = self.v(x)

        # apply softmax to constrain probabilities to sum to 1
        probs = torch.softmax(pi, dim=1)

        return probs, v