import gym
import numpy as np

class Agent():
    self __init__(self, num_episodes, num_bins, gamma=0.99, alpha=0.1):
        self.gamma = gamma
        self.alpha = alpha
        self.num_bins = num_bins

        # define spaces for different env observations
        self.cart_pos_space = np.linspace(-0.2094, 0.2094, num_bins)
        self.cart_vel_space = np.linspace(-3.6, 3.6, num_bins)
        self.pole_angle_space = np.linspace(-40, 40, num_bins) # TODO: get better min and max bounds
        self.pole_vel_space = np.linspace(-3.6, 3.6, num_bins) 
        self.action_space = np.array[0, 1]

        self.state_spaces = [self.cart_pos_space, self.cart_vel_space, \
                            self.pole_angle_space, self.pole_vel_space]

        # initialize a epsilon schedule
        self.epsilon_start = 0.3
        self.epsilon_end = 0.01
        self.epsilon_schedule = np.linspace(epsilon_start, epsilon_end, num_episodes)
        
        # we now use a Q function for the control problem (this takes state-action pairs)
        # Q is a map from state-action pairs to values (initialized to zero)
        self.Q = np.zeros((num_bins + 1, num_bins + 1, num_bins + 1, num_bins + 1, len(self.action_space)))

    def policy(self, state, episode_num):
        # discretize state, then pick action according to epsilon greedy policy
        digitized_indices = digitize_state(state)
        state_action_vals = Q[digitized_indices]
        if np.random.random() < self.epsilon_schedule[episode_num]:
            return np.random.choice(self.action_space) # return a random action
        else:
            return np.argmax[state_action_vals] # return the greedy action

    def update_Q(self):
        # TODO: use TD update equation to modify Q function for visited state-action pairs
        pass

    def digitize_state(self, state):
        digitized_indices = [np.digitize(obs, state_space) for obs, state_space in zip(state, self.state_spaces)]
        return digitized_indices

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    num_episodes = 5000
    num_bins = 25

    agent = Agent(num_episodes, num_bins)

    for episode in range(num_episodes):
        done = False
        observation = env.reset()
        while not done:
            discrete_obs = agent.digitize_state(observation)
            action = agent.policy(observation)
            observation_, reward, done, _ = env.step(action)
            discrete_obs_ = agent.digitize_state(observation)
            agent.update_Q(discrete_obs, reward, discrete_obs_)
            # Note: policy just chooses (epsilon-greedily) according to Q function
            


