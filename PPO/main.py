import gym
import sys
import getopt
import numpy as np
import time
from agent_class import Agent
from utils import plot_learning_curve

def train(env_name):
	env = gym.make(env_name)
	agent = Agent(alpha=0.0003, beta=0.001, gamma=0.99, input_shape=env.observation_space.shape,
					n_actions=env.action_space.n, fc1_dims=256, fc2_dims=256)
	n_games = 1500
	steps_per_update = 4000
	pi_update_iter = 80
	value_update_iter = 80

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
	
	env.close()
	filename = f'{env_name}_{n_games}_games'
	figure_file = f'plots/{filename}.png'
	plot_learning_curve(scores, figure_file)

def render_games(env_name):
	env = gym.make(env_name)
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
			time.sleep(0.01)
			score += reward
			observation = observation_
		print(f"Episode {i}, score: {score}")
	env.close()

if __name__ == '__main__':
	# parse input arguments
	arg_env_name = 'CartPole-v1'
	arg_render = False
	arg_help = f"{sys.argv[0]} -e <env_name> | use -r to render games from saved policy"

	try:
		opts, args = getopt.getopt(sys.argv[1:], "hre:", ["help", "render", 
		"env_name="])
	except:
		print(arg_help)
		sys.exit(2)

	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print(arg_help)  # print the help message
			sys.exit(2)
		elif opt in ("-e", "--env_name"):
			arg_env_name = arg
		elif opt in ("-r", "--render"):
			arg_render = True
	
	if arg_render:
		render_games(arg_env_name)
	else:
		train(arg_env_name)