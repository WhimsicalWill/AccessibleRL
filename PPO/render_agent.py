import gym
import numpy as np
import time
from agent_class import Agent

if __name__ == '__main__':
	env = gym.make('CartPole-v1')
	agent = Agent(alpha=0.0003, beta=0.001, gamma=0.99, input_shape=env.observation_space.shape,
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
			env.render(mode="human")
			time.sleep(0.03)
			score += reward
			observation = observation_
		print(f"Episode {i}, score: {score}")
	env.close()
