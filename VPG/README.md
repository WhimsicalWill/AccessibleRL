# Vanilla Policy Gradient

# Conceptual Overview

Vanilla Policy Gradients (VPG) is a classic RL algorithm. It is an on-policy method, meaning that that the policy used to collect experience is the same one that is updated. It can be used with discrete or continuous action spaces.

Taken from OpenAI's Spinning Up -- "The key idea underlying policy gradients is to push up the probabilities of actions that lead to higher return, and push down the probabilities of actions that lead to lower return, until you arrive at the optimal policy."

# Networks

- Actor network is a Deep NN with one hidden layer that maps states -> action distribution
- Value network is a Deep NN with one hidden layer that maps states -> value

Both the Actor and Critic use the same hidden layer to learn a representation that is useful for jointly learning the optimal actions and values for a particular environment.

# Learning Update

At each environment step, the `actor network` takes in the current state and outputs an action distribution. The agent samples from this distribution, and records the log_prob of the action taken.

After a single transition, the agent uses the transition (s, a, r, s') and the log_prob to update its parameters. 

First, the One-step TD target is calculated by adding the reward to the discounted estimate of the next state's value. We update the `value network` using temporal difference learning. This update rule nudges the evaluation of the current state towards the One-step TD target, which is a more informed estimate taken one time step in the future. We accomplish this update by performing gradient descent on the Mean Squared Error (MSE) between the current state value and the One-step TD target.

For the actor update, we use an `advantage function` to estimate if the action we took was better or worse than average. We formulate the advantage of a particular action as the `One-step TD target` - `current value`. If this quantity is positive, then the action was better than expected. Intuitively, actions that perform better than average should have their likelihood increased, and actions performing below average should have their likelihood decreased. To accomplish this, we maximize the quantity `log_prob` * `advantage`. Remember that the logarithm is a monotonic increasing function, so increasing the probability always increases the value of the `log_prob`.

# Other Information

- The Replay Buffer is reset after each learning update