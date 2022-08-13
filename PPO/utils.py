import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from agent_class import Agent

def plot_learning_curve(scores, figure_file):
	x = [i+1 for i in range(len(scores))]
	running_avg = np.zeros(len(scores))
	for i in range(len(running_avg)):
		running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
	plt.plot(x, running_avg)
	plt.title('Running average of previous 100 scores')
	plt.savefig(figure_file)

class ReplayBuffer:
	def __init__(self):
		self.clear()

	def clear(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.log_probs = []
		self.is_terminals = []

	def store_transition(self, state, action, reward, log_prob, is_terminal):
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.log_probs.append(log_prob)
		self.is_terminals.append(is_terminal)

def render_games(env_name):
	env = gym.make(env_name)
	agent = agent_class.Agent(alpha=0.0003, beta=0.001, gamma=0.99, input_shape=env.observation_space.shape,
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

