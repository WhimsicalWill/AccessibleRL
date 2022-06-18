# each thread can have a different agent class
# complete with:
# choose_action, store_transition, and learn functions
# also, each agent object has different networks

import numpy as np
from memory import Memory

class AgentProcess():
    def __init__(self, input_shape, n_actions, global_ac, optimizer):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.global_agent = global_ac # the global (central) controller
        self.optimizer = optimizer

        self.memory = Memory()
        self.actor_critic = ActorCritic(input_shape, n_actions)

    def store_transition(self, reward, value, log_prob):
        self.memory.store_transition(reward, value, log_prob)
    
    def choose_action(self, obs):
        state = torch.tensor([obs], dtype=torch.float) # batchify
        probs, value = self.actor_critic(state)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # no need for .cpu() before .numpy() as we are on cpu already
        return action.numpy()[0], v, log_prob

    def learn(self):
        # calculate rewards to go
        # then calculate GAE
        # then calculate losses
        # use backward(), then do optimizer step

        # load environment transitions that are used for gradient update
        rewards, values, log_probs = self.memory.sample_memory()

        optimizer.zero_grad()
        loss = self.calc_loss(obs, done, rewards, values, log_probs)
        loss.backward() # compute gradient of loss w.r.t. local agent's parameters
        torch.nn.utils.clip_grad_norm_(local_agent.parameters(), 40) # in-place gradient norm clip
        copy_gradients_and_step(local_agent, global_ac, optimizer)
        self.memory.reset() # clear the memory after a gradient update

    def copy_gradients_and_step(self, local_agent, global_ac, optimizer):
        for local_param, global_param in zip(actor_critic.parameters(), global_ac.parameters()):
            global_param.grad = local_param.grad
        optimizer.step()
        actor_critic.load_state_dict(global_ac.state_dict())