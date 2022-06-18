import gym
import numpy as np
from agent_class import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make("BipedalWalker-v3")
    agent = Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space.shape,
                    tau=0.005, env=env, batch_size=100, layer1_size=400, layer2_size=300, 
                    action_dim=env.action_space.shape[0])
    n_games = 1500
    filename = f'Walker2d_alpha_{agent.alpha}_beta_{agent.beta}_{n_games}_games'
    figure_file = f'plots/{filename}.png'

    best_score = env.reward_range[0] # init to smallest possible reward
    score_history = []
    for i in range(n_games): # Train TD3 Algorithm for many steps
        observation = env.reset()
        done = False
        score = 0
        while not done: # while current episode not finished
            action = agent.choose_action(observation) # choose action according to current policy
            observation_, reward, done, info = env.step(action) # environment interaction step
            agent.store_transition(observation, action, reward, observation_, done) # store transition
            agent.learn() # update networks according to TD3 algorithm
            score += reward
            observation = observation_
        score_history.append(score) # add score to list after episode
        avg_score = np.mean(score_history[-100:]) # average the last 100 episodes

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        print(f"Episode {i}, score: {score}, avg_score: {avg_score}")
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)

## Conceptual note: DDPG chases a much much more stable target_critic network
# While TD3 updates target networks every 2 steps with high (5x) tau
# DDPG, in contrast, updates every ~20 steps with very low tau
