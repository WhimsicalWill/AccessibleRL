import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
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

