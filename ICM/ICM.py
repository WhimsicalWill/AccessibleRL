import torch
import torch.nn as nn
import numpy as np

class ICM(nn.Module):
    def __init__(self, input_shape, n_actions, fc_size=256, alpha=0.1, beta=0.2):
        super(ICM, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta

        # self.conv1 = nn.Conv2D(input_shape[0], 32, 3, 3, stride=2, padding=1)
        # self.conv2 = nn.Conv2D(32, 32, 3, 3, stride=2, padding=1)
        # self.conv3 = nn.Conv2D(32, 32, 3, 3, stride=2, padding=1)
        # self.phi = nn.Conv2D(32, 32, 3, 3, stride=2, padding=1) # phi is our feature encoder

        # shape of latent features is (32, 3, 3)

        # take in two state representations and output logits for action space
        self.inverse_model = nn.Linear(288*2, fc_size)
        self.pi_logits = nn.Linear(fc_size, n_actions)

        # predict the next state from the current state and action
        self.forward_model = nn.Linear(288+1, fc_size)
        self.phi_hat_new = nn.Linear(288+1, fc_size)

        device = torch.device('cpu')
        self.to(device)

    def forward(self, states, new_states, actions):
        # calculate representations of states and new_states with feature encoder
        # calculate our predicted representation of the next state using state and action
        # with inverse model, predict action probs given state and new_state
        
        phi = self.phi((states))
        phi_new = self.phi((new_states))

        # Note: if we are working with a CNN encoder, we must broadcast feature rep to appropriate dimensions
        inverse = self.inverse_model(phi, phi_new)
        pi_logits = self.pi_logits(inverse)

        action = action.reshape(action.size[0], -1) # (B) -> (B, 1)
        forward_model = self.forward_model(torch.cat([states, actions], dim=1))
        phi_hat_new = self.phi_hat_new(forward_model)

        return phi_new, pi_logits, phi_hat_new