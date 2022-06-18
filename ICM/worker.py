import numpy  as np
import torch
import gym
from networks import ActorCritic
from memory import Memory
from utils import plot_learning_curve
    
def worker(name, input_shape, n_actions, global_agent, optimizer, env_id, n_threads, global_idx):
    T_MAX = 20
    local_agent = ActorCritic(input_shape, n_actions)
    memory = Memory()

    env = gym.make(env_id)
    
    t_steps = 0
    num_episodes = 1000
    scores = []
    for ep in range(num_episodes):
        obs = env.reset()
        score, done, ep_steps = 0, False, 0 # reset episode variables
        while not done:
            state = torch.tensor([obs], dtype=torch.float) # batchify the obs for our network
            action, value, log_prob = local_agent(state) # calls actorcritic forward function
            obs_, reward, done, _ = env.step(action)
            memory.store_transition(reward, value, log_prob)
            score += reward
            ep_steps += 1
            t_steps += 1
            obs = obs_
            if ep_steps % T_MAX == 0 or done: # update networks if needed
                rewards, values, log_probs = memory.sample_memory()
                # print(f"Values {values}")
                # print(f"logprobs {log_probs}")
                loss = local_agent.calc_loss(obs, done, rewards, values, log_probs)
                optimizer.zero_grad()
                loss.backward() # compute gradient of loss w.r.t. local agent's parameters
                torch.nn.utils.clip_grad_norm_(local_agent.parameters(), 40) # in-place gradient norm clip
                copy_gradients_and_step(local_agent, global_agent, optimizer)
                memory.reset() # clear the memory after a gradient update 
        if name == '1':
            scores.append(score)
            avg_score = np.mean(scores[:-100])
            print(f"Agent {name}: Episode {ep}, {score} score, {ep_steps} steps, avg score: {avg_score}")

    if name == '1': # plot learning curve for agent on first thread
        step_list = [x for x in range(num_episodes)]
        plot_learning_curve(step_list, scores, 'A3C_pong_final.png')

def copy_gradients_and_step(local_agent, global_agent, optimizer):
    for local_param, global_param in zip(local_agent.parameters(), global_agent.parameters()):
        global_param.grad = local_param.grad
    optimizer.step()
    local_agent.load_state_dict(global_agent.state_dict())