import torch
import torch.nn.functional as F
import numpy as np
from ou_noise import OUActionNoise
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                    max_size=1000000, fc1_dims=400, fc2_dims=300,  batch_size=64):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions, "actor")
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, "critic")
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # initialize target actor and critic (these don't learn, but merely lag behind actor/critic)
        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions, "target_actor")
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, "target_critic")

        self.update_agent_parameters(tau=1) # hard update with tau=1 for initial full copying of weights

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, state):
        self.actor.eval() # switch the NN into evaluation mode
        state = torch.tensor([state], dtype=torch.float).to(self.actor.device) # put on GPU
        mu = self.actor(state).to(self.actor.device) # gets a normal distribution centered at best action
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)
        self.actor.train() # switch back into training mode

        return mu_prime.cpu().detach().numpy()[0] # TODO: why [0]

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return # don't learn until we can sample at least a full batch

        # Sample memory buffer uniformly
        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

        # Convert from numpy arrays to torch tensors for computation graph
        states = torch.tensor([states], dtype=torch.float).to(self.actor.device)
        actions = torch.tensor([actions], dtype=torch.float).to(self.actor.device)
        states_ = torch.tensor([states_], dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done, dtype=torch.float).to(self.actor.device)
        
        # Update the critic to minimize the MSE w.r.t. the target critic
        self.critic.optimizer.zero_grad()
        target_actions = self.target_actor(states_)

        # TODO: delete shape manipulation redundancy
        target_q = rewards + (1.0 - done) * (self.gamma * self.target_critic(states_, target_actions).view(-1))
        target_q = target_q.view(self.batch_size, 1) # shape (B, 1)
        critic_loss = F.mse_loss(self.critic(states, actions), target_q)
        critic_loss.backward()
        self.critic.optimizer.step()

        # Update the actor greedily w.r.t to the target critic
        self.actor.optimizer.zero_grad()
        actor_output = self.actor(states)
        actor_loss = -torch.mean(self.target_critic(states, actor_output))
        actor_loss.backward()
        self.actor.optimizer.step()

        # Do a soft update after a learning step
        self.update_agent_parameters()

    def update_agent_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        self.update_model_parameters(tau, self.critic, self.target_critic)
        self.update_model_parameters(tau, self.actor, self.target_actor)

    def update_model_parameters(self, tau, model, target_model): # TODO: find difference btwn named_params and params
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            updated_param = tau * param.data + (1 - tau) * target_param.data
            target_param.data.copy_(updated_param) # update the target's weights

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
