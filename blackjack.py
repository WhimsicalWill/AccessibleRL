import numpy as np

class Agent():
    def __init__(self, gamma=0.99):
        self.states = {} # dict mapping from a state's tuple description to State object
        self.sum_space = list(range(4, 22))
        self.dealer_show_card_space = list(range(1, 11))
        self.ace_space = [False, True]
        self.action_space = [0, 1]

        self.state_space = []
        self.memory = []
        self.gamma = gamma

        self.init_vals()

    # note: we could store state data in tensor by just converting all state components to ints
    def init_vals(self):
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    self.states[(total, card, ace)] = State([], False, 0)
                    self.state_space.append((total, card, ace)) # don't know what this is for

    # define agent's policy given a state
    def policy(self, state):
        total, _, _ = state # our initial policy is based merely on player's total
        action = 0 if total >= 20 else 1
        return action
    
    # update w/ monte carlo according to value of first visit O(n^2)
    def update_V(self):
        for i, (state, _) in enumerate(self.memory):
            G = 0 # total reward
            if not self.states[state].visited:
                self.states[state].visited = True
                discount = 1 # init discount to 1
                for t, (_, reward) in enumerate(self.memory[i:]):
                    G += discount * reward
                    discount *= self.gamma
                self.states[state].returns.append(G) # add G to the list of returns from this state
                self.states[state].value = np.mean(self.states[state].returns) # update value for state

        # reset each state's visited field to False
        for state in self.state_space: 
            self.states[state].visited = False # could be moved to above loop for efficiency
        self.memory = [] # reset memory


class State():
    def __init__(self, returns=[], visited=False, value=0):
        self.returns = returns
        self.visited = visited
        self.value = value