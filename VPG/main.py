import gym
import numpy as np
from agent_class import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
	env = gym.make('CartPole-v1')
	agent = Agent(gamma=0.99, lr=5e-6, input_dims=env.observation_space.shape,
					n_actions=env.action_space.n, fc1_dims=256, fc2_dims=256)
	n_games = 5000
	filename = f'CartPole_{n_games}_games'
	figure_file = f'plots/{filename}.png'

	best_score = env.reward_range[0] # init to smallest possible reward
	scores = []
	for i in range(n_games):
		done = False
		observation = env.reset()
		score = 0
		while not done:
			action, log_prob = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			score += reward
			agent.learn(observation, reward, observation_, log_prob, done)
			observation = observation_
		scores.append(score)
		avg_score = np.mean(scores[-100:])

		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()
		print(f"Episode {i}, score: {score}, avg_score: {avg_score}")
	plot_learning_curve(scores, figure_file)

