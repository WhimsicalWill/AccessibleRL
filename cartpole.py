import numpy as np
import gym

class Agent():
    def __init__(self, num_bins=10, gamma=0.99, alpha=0.1):
        self.states = {} # states dict
        self.cart_position_space = np.linspace(-2.4, 2.4, num=num_bins)
        self.cart_velocity_space = [False, True]
        self.pole_angle_space = np.linspace(-41.8, 41.8, num=num_bins)
        self.pole_velocity_space = [False, True]

        self.state_space = []
        self.memory = []
        self.gamma = gamma
        self.alpha = alpha # learning rate (small leads to good convergence guarantees)

        self.init_vals()

    def init_vals(self): # init each state with a value
        for cart_pos in cart_position_space:
            for cart_vel in cart_velocity_space:
                for pole_angle in pole_angle_space:
                    for pole_vel in pole_velocity_space:
                        self.state_space.append((cart_pos, cart_vel, pole_angle, pole_vel))
                        for action in (0, 1):
                            self.states[(cart_pos, cart_vel, pole_angle, pole_vel)] = StateActionPair(0)

    def policy(self, state): # very simple policy
        _, _, pole_angle, _ = state
        if pole_angle < 0:
            return 0
        else:
            return 1 
    
    def update_Q(self):
        for t, (state, reward) in enumerate(self.memory[:-1]): # exclude last
            current_state_val = self.states[state].value
            next_state_val = self.states[self.memory[t+1]].value # use bootstrapping formula to update state estimates
            self.states[state].value = self.states[state].value + self.alpha(next_state + reward - current_state_val)
        
        # after updating Q value for each state in our memory, reset memory
        memory = []
            

class StateActionPair():
    def __init__(self, value):
        self.value = value

    