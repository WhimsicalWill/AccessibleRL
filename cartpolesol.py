import numpy as np
import gym

def simple_policy(state): # return action given state
    action = 0 if state < 5 else 1
    return action

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    alpha = 0.1
    gamma = 0.99

    num_bins = 10
    states = np.linspace(-0.2094, 0.2094, num_bins) # 10 dividers -> 11 different bins
    V = {}
    for state in range(len(states) + 1): # state keys are integers
        V[state] = 0 # value function maps states to values (init to zero)

    for i in range(5000):
        observation = env.reset()
        done = False
        while not done:
            state = int(np.digitize(observation[2], states)) # map continuous obs to a bin
            action = simple_policy(state)
            observation_, reward, done, info = env.step(action)
            state_ = int(np.digitize(observation_[2], states))
            V[state] = V[state] + alpha*(reward + gamma*V[state_] - V[state])
            observation = observation_
    
    # after training, print the state value estimates
    for state in V:
        print(state, '%.3f' % V[state])
