# Twin-Delayed Deep Deterministic Policy Gradients

# Conceptual Overview

Twin-Delayed Deep Deterministic Policy Gradients (TD3) is a further iteration of DDPG which mainly aims to prevent the overestimation bias inherent to function approximators. The core algorithm is the same as DDPG, except for a few key details.

TD3 updates two critic networks independently and maintains a `target_critic` for each one. During the learning update, it takes the minimum of the predicted Q values, and uses that to compute the One-step TD targets. 

The algorithm also has a short 'warmup' phase where it takes actions sampled from a normal distribution to collect lots of experience and begin to train the networks before sampling actions from the learnable actor network.

It also gets rid of the OU Noise, in favor of exploration noise sampled from a normal distribution.

Finally, the actor network is updated at a slower rate than the critic network. Instead of every iteration, it is updated every D steps. The `target_actor`, `target_critic_1`, and `target_critic_2` are also updated every D steps, so they provide more stable learning targets. 

# Networks

- Actor is a Deep NN with one hidden layer that maps states -> actions
- Critic_1 is a Deep NN with one hidden layer that maps (state, action) -> value
- Critic_2 is a Deep NN with one hidden layer that maps (state, action) -> value

DDPG also has a `target_actor`, `target_critic_1`, and `target_critic_2` which are frozen copies of the actual `actor` and `critic` networks that lag slowly behind them and provide a stable learning target for both networks.

# Learning Update

A batch of (s, a, r, s) transitions are sampled from the replay buffer uniformly.

One-step TD targets are computed for each transition. The TD targets are computed as the reward added to the discounted `target_critic` value of the next state action pair. One caveat is that this discounted target critic value is now taken from the minimum of the two critics to curb overestimation bias. Since we do not have access to `a'` in our (s, a, r, s') transition, we use the `target_actor` to compute the action (`a'`) to be taken from the next state.

The critic loss is formulated as the Mean Squared Error (MSE) between the critic's predictions and the One-step TD Targets. A gradient step is taken for both critic networks in the direction that minimizes this loss across the whole batch of transitions.

For the actor update, we use the critic as a proxy that tells us which parts of the environment are high value. We feed our actor model's action into the `target_critic` (arbitrarily, we choose `target_critic_1`), and take a gradient step in the direction that maximizes the average `target_critic_1` values across the whole batch of transitions. This is the gradient of the predicted value w.r.t. the actor model's parameters, so the gradient step only modifies the actor's parameters.

Then we update our `target_actor`, `target_critic_1`, and `target_critic_2` which lag behind the actual `actor`, `critic_1`, and `critic_2`. This is implemented by taking an `exponentially weighted average` of the past parameters of the models.


# Other Information

- Uses a replay buffer
- Uses exploration noise sampled from a normal distribution