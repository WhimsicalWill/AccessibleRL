import gym
import numpy as np
from agent_class import Agent

if __name__ == '__main__':
	env = gym.make('CartPole-v1')
	agent = Agent(gamma=0.99, lr=5e-6, input_dims=env.observation_space.shape,
					n_actions=env.action_space.n, fc1_dims=256, fc2_dims=256)
	n_games = 10

	# Load saved model
	agent.load_models()

	for i in range(n_games):
		observation = env.reset()
		done = False
		score = 0
		while not done:
			action, _ = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			env.render()
			score += reward
			observation = observation_
		print(f"Episode {i}, score: {score}")
