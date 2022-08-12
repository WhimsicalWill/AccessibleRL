import gym
import numpy as np
from agent_class import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
	env = gym.make('CartPole-v1')
	agent = Agent(alpha=0.0003, beta=0.001, gamma=0.99, input_dims=env.observation_space.shape,
					n_actions=env.action_space.n, fc1_dims=256, fc2_dims=256)
	n_games = 100
	steps_per_update = 4000
	pi_update_iter = 80
	value_update_iter = 80
 
 
	filename = f'CartPole_{n_games}_games'
	figure_file = f'plots/{filename}.png'

	best_score = env.reward_range[0] # init to smallest possible reward
	scores = []
	steps = 0
	for i in range(n_games):
		done = False
		observation = env.reset()
		score = 0
		while not done:
			action, log_prob = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			agent.store_transition(observation, action, reward, log_prob, done)
			score += reward
			steps += 1
			if steps % steps_per_update == 0:
				agent.learn(pi_update_iter, value_update_iter)
			observation = observation_
		scores.append(score)
		avg_score = np.mean(scores[-100:])

		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()
		print(f"Episode {i}, score: {score}, avg_score: {avg_score}")
	x = [i+1 for i in range(n_games)]
	plot_learning_curve(x, scores, figure_file)

