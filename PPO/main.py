import gym
import numpy as np
from agent_class import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
	# Collect trajectories using policy Theta_k
	time_step = 0
	episode = 0
	while time_step < self.max_time_steps:
		state = env.reset()

		if episode % (100) == 0:
			self.render = True
		else:
			self.render = False
		episode += 1

		episode_reward = 0 # finite time horizons
		for i in range(self.max_ep_length):
			action = self.select_action(state)
			new_state, reward, done, _ = env.step(action)
			time_step += 1
			episode_reward += reward

			if self.render:
				env.render()
				# time.sleep(0.01)

			# add data
			self.data.rewards.append(reward)
			self.data.is_terminals.append(done)

			# update if needed
			if time_step % self.samples_per_policy == 0:
				self.update()

			if done:
				break

			state = new_state
		print("max episode step reached:", i, " with reward:", episode_reward)
		self.reward_history.append(episode_reward)
