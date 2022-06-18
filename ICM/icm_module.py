import torch
import torch.nn as nn
import numpy as np

class ICM(nn.Module):
    def __init__(self, input_shape, n_actions, fc_size=256):
        super(ICM, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.conv1 = nn.Conv2D(input_shape[0], 32, 3, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2D(32, 32, 3, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2D(32, 32, 3, 3, stride=2, padding=1)
        self.phi = nn.Conv2D(32, 32, 3, 3, stride=2, padding=1) # phi is our feature encoder

        # shape of latent features is (32, 3, 3)

        self.inverse_model = nn.Linear(288*2, fc_size)
        self.pi_logits = nn.Linear(fc_size, n_actions)

        self.dense = nn.Linear(288+1, fc_size)
        self.phi_hat_new = nn.Linear(288+1, fc_size)

        device = torch.device('cpu')
        self.to(device)

    def forward(self, state):
        pass

    def calc_loss(self, state, action, new_state):
        pass