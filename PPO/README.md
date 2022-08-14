# Proximal Policy Optimization

# Conceptual Overview

Proximal Policy Optimization (PPO) ensures that policy updates do not drastically change the policy during each update, and this leads to better stability and convergence on the optimal policy. There are two mainstream implementations of PPO, one which penalizes the actor based on KL divergence, and one which clips the policy. This implementation implements the latter, also known as PPO-Clip.

PPO is similar to Vanilla Policy Gradients (VPG), and we will see that its loss function takes a very similar form, except for the explicit clipping employed by PPO. In PPO, we run gradient descent for many timesteps whenever we update our `actor` and `value` networks. During this update, we use a small constant `epsilon` to control how far our new policy can stray from the old one. Once we increase or decrease a given action's log_probs past a certain extent, we will get gradients of zero and this action will no longer contribute to the gradient of the actor_loss.

Like VPG, PPO is on-policy and works for continuous and discrete environments. PPO keeps track of (s, a, r, s') transitions as well as the `log_prob` of the actions taken under the current policy.

In this implementation, we also include a hyperparameter `entropy_weight` to control the contribution of an entropy bonus to the loss. This is to encourage exploration and prevent premature convergence to local optima, but can be disabled by setting the hyperparameter to 0.

# Networks

- Actor is a Deep NN with one hidden layer that maps states -> action distribution
- Value is a Deep NN with one hidden layer that maps states -> value

# Learning Update

Because PPO is an on-policy method, it clears the replay buffer after each learning update in order to strictly use experience collected under the current policy to update.

All transitions from the episode, having form (s, a, r, s'), along with the log_probs corresponding to the actions, are loaded into torch tensors.

Rewards-to-go are calculated by summing the discounted rewards of the episode, and taking into account the fact that a terminal state has a reward of zero. These rewards-to-go are then normalized.

Since we want to prevent our new policy from deviating too far from our old policy on each learning update, we ensure that the ratio of log_probs (between the old and new policy) does not exceed `1-epsilon` or `1+epsilon`. We then calculate our `actor_loss` as the `ratios` * `advantages`

We optimize the `actor_loss` by nudging the actor parameters in the direction of the gradient. However, since our loss function includes the log_probs of the old policy, we detach them to reflect the fact that the old policy's log_probs are static throughout all the iterations of gradient descent.

The policy uses gradient descent for a large amount of iterations -- this implementation uses 80 iterations of gradient descent for both the policy update and value update.

The `value_loss` is formulated as the Mean Squared Error (MSE) between the rewards-to-go and the `state_values` returned by the `value network`. We minimize this loss for 80 iterations using gradient descent.

After the update, we clear the replay buffer and continue to collect more experience.

# Other Information

- On policy method
- Multiple iterations of gradient descent at each update
- Rewards-to-go are normalized

# TODO

- Change Actor to Value to reflect that it has state values, not state-action vals
- Try to work in the entropy bonus
- Document tensor shapes at intermediate steps
