class RolloutBuffer:
	def __init__(self):
		self.clear()

	def clear(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.log_probs = []
		self.is_terminals = []
