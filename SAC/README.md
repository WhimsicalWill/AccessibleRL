# Soft Actor Critic

# Conceptual Overview

Soft Actor Critic (SAC) is an `off-policy` RL algorithm that uses a `stochastic policy` and works with environments that have `continuous action spaces`. It learns two independent critic networks to curb overestimation bias, and is similar to `DDPG` and `TD3`. Much of SAC's stability comes from its stochastic nature and `entropy regularization`.

Taken from OpenAI's Spinning Up docs: "A central feature of SAC is entropy regularization. The policy is trained to maximize a trade-off between expected return and entropy, a measure of randomness in the policy. This has a close connection to the exploration-exploitation trade-off: increasing entropy results in more exploration, which can accelerate learning later on. It can also prevent the policy from prematurely converging to a bad local optimum."

# Networks

- Actor is a Deep NN with one hidden layer that maps states -> normal distribution on actions
- Critic is a Deep NN with one hidden layer that maps (state, action) -> value
- ValueNetwork is a Deep NN with on ehidden layer mapping states -> values

SAC also has a `target_value` which is a frozen copy of the actual `value` network that lags slowly behind it and provides a stable learning target.

Note: not all implementations of SAC have a separate Value function, but this implementation does.

# Learn Function

A batch of (s, a, r, s') transitions are sampled from the replay buffer uniformly.

The three networks are updated in the learn function, but keep in mind that the order that they are updated is arbitrary and inconsequential. This implementation updates the `value` network, `actor`, and then finally the `critic`

The `value` network's loss is formulated as the Mean Squared Error (MSE) between the `value` network's prediction of the state values and the `value_target` which approximates the expected value of the states plus the `entropy regularization`. A gradient step is taken in the space of the `value` networks parameters in order to minimize this MSE loss.

For its update, the `actor` uses the minimum value of both critics as a proxy that tells it which parts of the environment are high value. We feed our actor model's action into both `critic` networks, and maximize the minimum value of the two networks plus the actor's bonus for acting with higher uncertainty. Then we take  take a gradient step in the direction that maximizes this metric across the whole batch of transitions. 

The `critic` loss is formulated as the MSE between the critic's predictions and the One-step TD Targets. These TD Targets are computed as the next-step reward added to the discounted `target_value` network's value prediction of the next state. A gradient step is taken in the direction that minimizes this loss across the whole batch of transitions.

Then we update our `target_value` network which lags behind the actual `value` network. This is implemented by taking an `exponentially weighted average` of the past parameters of the `value` network.

# Other Information

- Uses a replay buffer
- This implementation uses a Value Network
- The stochasticity of the policy and the entropy regularization takes care of exploration.
- Actions are sampled from a normal during training, but the mean action is taken during evaluation time.
