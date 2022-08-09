# Soft Actor Critic

# Conceptual Overview

Deep Deterministic Policy Gradient (DDPG) is the continuous analog of Deep Q-Network (DQN). Instead of producing discrete actions, DDPG produces continuous-valued actions.

The main practical issue that DDPG addresses is the problem of finding the best action in a continuous space. In the discrete setting, a critic function makes the action policy trivial. It must simply look at the critic's evaluation of each (state, action) pair, and take the action with the highest predicted value. However, the equivalent in a continuous setting is impossible.

In contrast to DQN, DDPG is an Actor Critic algorithm. This implementation uses Deep NN function approximators for choosing a continuous action vector (`actor`) and for  evaluating (state, action) pairs (`critic`).

# Networks

- Actor is a Deep NN with one hidden layer that maps states -> actions
- Critic is a Deep NN with one hidden layer that maps (state, action) -> value

DDPG also has a `target_actor` and `target_critic` which are frozen copies of the actual `actor` and `target` that lag slowly behind them and provide a stable learning target for both networks.

# Learn Function

A batch of (s, a, r, s) transitions are sampled from the replay buffer uniformly.

One step TD targets are computed for each transition. The target is computed as the reward added to the discounted target critic value of the next state action pair. Since we only have access to the resultant state, we use the target_actor to compute the action.

The critic loss is formulated as the Mean Squared Error (MSE) between the critic's predictions and the One-step TD Targets. A gradient step is taken in the direction that minimizes this loss across the whole batch of transitions.

For the actor update, we use the critic as a proxy that tells us which parts of the environment are high value. We feed our actor model's action into the target critic, and take a gradient step in the direction that maximizes the average critic values across the whole batch of transitions. This is the gradient of the predicted value w.r.t. the actor model's parameters, so the gradient step only modifies the actor's parameters.

Then we update our `target_actor` and `target_critic` which lag behind the actual `actor` and `critic`. This is implemented by taking an `exponentially weighted average` of the past parameters of the `actor` and `critic`


# Other Information

- Uses a replay buffer
- Uses Ornstein Uhlenbeck noise (OU Noise)
    - This is correlated noise that is added to actions to induce exploration, and has a tendency to drift back towards zero.
- Networks use a specific weight initialization from the DDPG paper
