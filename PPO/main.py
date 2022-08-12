import gym
import numpy as np
from agent_class import Agent
from utils import plot_learning_curve

# if __name__ == '__main__':
# 	# Collect trajectories using policy Theta_k
# 	time_step = 0
# 	episode = 0
# 	while time_step < self.max_time_steps:
# 		state = env.reset()

# 		if episode % (100) == 0:
# 			self.render = True
# 		else:
# 			self.render = False
# 		episode += 1

# 		episode_reward = 0 # finite time horizons
# 		for i in range(self.max_ep_length):
# 			action = self.select_action(state)
# 			new_state, reward, done, _ = env.step(action)
# 			time_step += 1
# 			episode_reward += reward

# 			if self.render:
# 				env.render()
# 				# time.sleep(0.01)

# 			# add data
# 			self.data.rewards.append(reward)
# 			self.data.is_terminals.append(done)

# 			# update if needed
# 			if time_step % self.samples_per_policy == 0:
# 				self.update()

# 			if done:
# 				break

# 			state = new_state
# 		print("max episode step reached:", i, " with reward:", episode_reward)
# 		self.reward_history.append(episode_reward)

import gym
import numpy as np
from agent_class import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
	env = gym.make('CartPole-v1')
	agent = Agent(gamma=0.99, lr=5e-6, input_dims=env.observation_space.shape,
					n_actions=env.action_space.n, fc1_dims=256, fc2_dims=256)
	n_games = 5000
	steps_per_update = 4000
	pi_update_iter = 80
	value_update_iter = 80
 
 
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
			agent.store_transition(observation, action, reward, log_prob, done)
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

