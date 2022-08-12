import torch
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                    update_actor_interval=2, warmup=1000, action_dim=2,
                    max_size=1000000, layer1_size=400, layer2_size=300,
                    batch_size=100, noise=0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.noise = noise

        self.min_action = env.action_space.low
        self.max_action = env.action_space.high

        self.time_step = 0
        self.learn_step_cntr = 0
        self.warmup = warmup
        self.update_actor_interval = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, action_dim, "actor")
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, action_dim, "critic_1")
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, action_dim, "critic_2")
        self.memory = ReplayBuffer(max_size, input_dims, action_dim)

        # initialize target networks (these don't learn, but merely lag behind actor/critic)
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, action_dim, "target_actor")
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, action_dim, "target_critic_1")
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, action_dim, "target_critic_2")

        self.update_agent_parameters(tau=1) # hard update with tau=1 for initial full copying of weights

    def choose_action(self, observation):
        if self.time_step < self.warmup: # random action
            mu = torch.normal(0, self.noise, (self.action_dim,)).to(self.actor.device)
        else: # deterministic action
            state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
            # TODO: must we add a batch dimensions for states going to the actor?
            mu = self.actor(state) # TODO: is putting onto cuda device not necessary? (model and state are already on it)
        mu_prime = mu + torch.normal(0, self.noise, (self.action_dim,)).to(self.actor.device)
        mu_prime = torch.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_ctr < self.batch_size:
            return # don't learn until we can sample at least a full batch

        # Sample memory buffer uniformly
        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

        # Convert from numpy arrays to torch tensors for computation graph
        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done, dtype=torch.float).to(self.actor.device)
        
        # <---- CRITIC UPDATE ---->
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        target_actions = self.target_actor(states_) # states on gpu -> target_actions on gpu after call
        clipped_noise = torch.clip(torch.normal(0, 0.2, (1,)), -0.5, 0.5).to(self.actor.device) # put on the device
        target_actions = torch.clamp(target_actions + clipped_noise, self.min_action[0], self.max_action[0]) # not clipping this may result in OOD function approx.

        # take minimum of both target networks estimates to curb overstimation bias
        target_q1 = rewards + (1.0 - done) * (self.gamma * self.target_critic_1(states_, target_actions).view(-1))
        target_q2 = rewards + (1.0 - done) * (self.gamma * self.target_critic_2(states_, target_actions).view(-1))
        target_q = torch.min(target_q1, target_q2) # torch.min takes element-wise minimum
        target_q = target_q.view(self.batch_size, 1) # shape (B, 1) 
        critic_loss_1 = F.mse_loss(self.critic_1(states, actions), target_q)
        critic_loss_2 = F.mse_loss(self.critic_2(states, actions), target_q)
        (critic_loss_1 + critic_loss_2).backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        # <---- ACTOR UPDATE ---->
        if self.learn_step_cntr % self.update_actor_interval == 0: # update actor every D timesteps      
            self.actor.optimizer.zero_grad()
            actor_output = self.actor(states)
            actor_loss = -torch.mean(self.critic_1(states, actor_output)) # Update the actor greedily w.r.t to the target critic
            actor_loss.backward()
            self.actor.optimizer.step()
            self.update_agent_parameters() # Do a soft update after a learning step

    # Not sure if this hard parameter will work since it references a class field
    def update_agent_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        self.update_model_parameters(tau, self.critic_1, self.target_critic_1)
        self.update_model_parameters(tau, self.critic_2, self.target_critic_2)
        self.update_model_parameters(tau, self.actor, self.target_actor)

    def update_model_parameters(self, tau, model, target_model): # TODO: find difference btwn named_params and params
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            updated_param = tau * param.data + (1 - tau) * target_param.data
            target_param.data.copy_(updated_param) # update the target's weights

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_2.load_checkpoint()
