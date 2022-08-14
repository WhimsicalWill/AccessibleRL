# Vanilla Policy Gradient

# Conceptual Overview

Vanilla Policy Gradients (VPG) is a classic RL algorithm. It is an on-policy method, meaning that that the policy used to collect experience is the same one that is updated. It can be used with discrete or continuous action spaces.

Taken from OpenAI's Spinning Up -- "The key idea underlying policy gradients is to push up the probabilities of actions that lead to higher return, and push down the probabilities of actions that lead to lower return, until you arrive at the optimal policy."

At each environment step, the `actor network` takes in the current state and outputs an action distribution. The agent samples from this distribution, and records the state, action, reward, and whether or not the state is terminal to the replay buffer.

# Networks

- Actor network is a Deep NN with one hidden layer that maps states -> action distribution
- Value network is a Deep NN with one hidden layer that maps states -> value

# Learning Update

All transitions from the episode, having form (s, a, r, s'), are loaded into torch tensors. We also store the `critic network`'s prediction of the state evalutaions, and the `log_probs` of the actions taken under the current policy of the `actor network`.

Rewards-to-go are calculated by summing the discounted rewards of the episode, and taking into account the fact that a terminal state has a reward of zero. These rewards-to-go are then normalized.

For the actor update, we use an `advantage function` to estimate if the action we took was better or worse than average. We formulate the advantage of a particular action as the `rewards-to-go - current value`. If this quantity is positive, then the action was better than expected. Intuitively, actions that perform better than average should have their likelihood increased, and actions performing below average should have their likelihood decreased. To accomplish this, we maximize the quantity `log_prob * advantage`. Remember that the logarithm is a monotonic increasing function, so increasing the probability always increases the value of the `log_prob`.

The `value_loss` is formulated as the Mean Squared Error (MSE) between the rewards-to-go and the `state_values` returned by the `value network`. We minimize this MSE loss by taking a step in the direction of the gradient.

After the update, we clear the replay buffer and continue to collect more experience.

# Other Information

- The Replay Buffer is reset after each learning update
- Log_probs are not stored, but calculated at update time