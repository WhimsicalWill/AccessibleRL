class RolloutBuffer:
	def __init__(self):
		self.clear()

	def clear(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.log_probs = []
		self.is_terminals = []

	def store_transition(self, state, action, reward, log_prob, is_terminal):
		self.state.append(state)
		self.action.append(action)
		self.reward.append(reward)
		self.log_prob.append(log_prob)
		self.is_terminal.append(is_terminal)

