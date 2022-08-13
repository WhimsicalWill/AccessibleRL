# Deep Q-Network

# Conceptual Overview

Deep Q-Network (DQN) approximates the value of taking different actions in each state of the environment through the use of a neural network or some other function approximator.

It is an off-policy method that explores the environment by sometimes taking random actions. This is known as epsilon-greedy exploration, and some implementations may use an epsilon schedule to decrease the random action probability as learning progresses.

When our agent does not take random actions to explore, it predicts the action values of each possible action in the given state and executes the action with the highest predicted value.

The Q-network, which predicts the value of taking an action in a given state, is trained by nudging `Q(s,a)` towards `reward + gamma * max_a(Q(s',a))`. Since we nudge towards a target that greedily takes the best action instead of acting randomly with probability epsilon, DQN is considered off-policy.

# Networks

- Critic is a Deep NN with one hidden layer that maps (state) -> action values

# Learning Update

A batch of (s, a, r, s') transitions are sampled from the replay buffer uniformly.

To update our `critic network` which predicts the future discounted rewards of taking each action in a given state (the Q-values), we calculate a target using the reward received at each transition plus the discounted Q-value of the best action in the next state. If the next state was terminal, the target is just the reward.

`target = reward + (1 - is_terminal) * gamma * max_a(Q(s', a))`

Then we grab the `critic network`'s prediction of the action taken in the current state `Q(s, a)` and minimize the Mean Squared Error (MSE) betwen this prediction and the target using gradient descent. 

# Other Information

- Uses a replay buffer