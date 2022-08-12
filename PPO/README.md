# Proximal Policy Optimization

# Conceptual Overview

Proximal Policy Optimization (PPO) builds off of Trust Region Policy Optimization (TRPO) in that it ensures that policy updates do not drastically change the policy at each step.

# Networks

- Actor is a Deep NN with one hidden layer that maps states -> actions
- Critic is a Deep NN with one hidden layer that maps (state, action) -> value

DDPG also has a `target_actor` and `target_critic` which are frozen copies of the actual `actor` and `critic` that lag slowly behind them and provide a stable learning target for both networks.

# Learning Update

A batch of (s, a, r, s') transitions are sampled from the replay buffer uniformly.

One-step TD targets are computed for each transition. The TD targets are computed as the reward added to the discounted `target_critic` value of the next state action pair. Since we do not have access to `a'` in our (s, a, r, s') transition, we use the `target_actor` to compute the action (`a'`) to be taken from the next state.

The critic loss is formulated as the Mean Squared Error (MSE) between the critic's predictions and the One-step TD Targets. A gradient step is taken in the direction that minimizes this loss across the whole batch of transitions.

For the actor update, we use the critic as a proxy that tells us which parts of the environment are high value. We feed our actor model's action into the `target_critic`, and take a gradient step in the direction that maximizes the average `target_critic` values across the whole batch of transitions. This is the gradient of the predicted value w.r.t. the actor model's parameters, so the gradient step only modifies the actor's parameters.

Then we update our `target_actor` and `target_critic` which lag behind the actual `actor` and `critic`. This is implemented by taking an `exponentially weighted average` of the past parameters of the `actor` and `critic`


# Other Information

- On policy method
- Multiple iterations of gradient descent at each update

# TODO

- Change Actor to Value to reflect that it has state values, not state-action vals
- Try to work in the entropy bonus
- Document tensor shapes at intermediate steps
