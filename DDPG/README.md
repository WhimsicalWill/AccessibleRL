# Deep Deterministic Policy Gradients

# Conceptual Overview

Deep Deterministic Policy Gradient (DDPG) is the continuous analog of Deep Q-Network (DQN). Instead of producing discrete actions, DDPG produces continuous-valued actions.

The main practical issue that DDPG addresses is the problem of finding the best action in a continuous space. In the discrete setting, a critic function makes the action policy trivial. It must simply look at the critic's evaluation of each (state, action) pair, and take the action with the highest predicted value. However, the equivalent in a continuous setting is impossible.

In contrast to DQN, DDPG is an Actor Critic algorithm. This implementation uses Deep NN function approximators for choosing a continuous action vector (`actor`) and for  evaluating (state, action) pairs (`critic`).

# Networks

- Critic is a Deep NN with one hidden layer that maps (state, action) -> value

# Learning Update

A batch of (s, a, r, s') transitions are sampled from the replay buffer uniformly.

One-step TD targets are computed for each transition. The TD targets are computed as the reward added to the discounted `target_critic` value of the next state action pair. Since we do not have access to `a'` in our (s, a, r, s') transition, we use the `target_actor` to compute the action (`a'`) to be taken from the next state.

# Other Information

- Uses a replay buffer