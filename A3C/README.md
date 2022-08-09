# Asynchronous Advantage Actor Critic

# Conceptual Overview

Asynchronous Advantage Actor Critic (A3C) is an on-policy method that overcomes many of the shortcomings of on-policy methods by collecting data from many processes in parallel. This implementation operates on environments with discrete action spaces, so it can be effectively thought of as a distributed version of Deep Q-Network (DQN).

It maintains a centralized policy and has a set amount of agents collecting data with their own local policy. After a given amount of time steps, the gradients of the local policy are calculated and sent back to the centralized policy to update it. Then, the local policy grabs a copy of the new centralized policy, effectively syncing with the  updates of the other local policies.

Instead of collecting transitions, the agent stores `rewards`, `values` and `log_probs` in a replay buffer. 

The loss used for A3C is a combination of the `actor loss`, `critic loss` and `entropy bonus`. The entropy bonus encourages the actor to maximize the uncertainty of its policy, which enables it to learn more effectively and not get stuck in local optima.

# Networks

- Actor is a Deep NN with one hidden layer that maps states -> actions
- Critic is a Deep NN with one hidden layer that maps (state, action) -> value

This implementation uses the same hidden layer to learn representations for both the Actor and Critic, which are two different output heads.

# Learning Update


Generalized Advantage Estimate

Replay buffer is cleared after update.


Transitions in the form of (s, a, r, s') are added to the agent

A batch of (s, a, r, s') transitions are sampled from the replay buffer uniformly.

One-step TD targets are computed for each transition. The TD targets are computed as the reward added to the discounted `target_critic` value of the next state action pair. One caveat is that this discounted target critic value is now taken from the minimum of the two critics to curb overestimation bias. Since we do not have access to `a'` in our (s, a, r, s') transition, we use the `target_actor` to compute the action (`a'`) to be taken from the next state.

The critic loss is formulated as the Mean Squared Error (MSE) between the critic's predictions and the One-step TD Targets. A gradient step is taken for both critic networks in the direction that minimizes this loss across the whole batch of transitions.

For the actor update, we use the critic as a proxy that tells us which parts of the environment are high value. We feed our actor model's action into the `target_critic` (arbitrarily, we choose `target_critic_1`), and take a gradient step in the direction that maximizes the average `target_critic_1` values across the whole batch of transitions. This is the gradient of the predicted value w.r.t. the actor model's parameters, so the gradient step only modifies the actor's parameters.

Then we update our `target_actor`, `target_critic_1`, and `target_critic_2` which lag behind the actual `actor`, `critic_1`, and `critic_2`. This is implemented by taking an `exponentially weighted average` of the past parameters of the models.


# Other Information

- Local agents use a shared adam optimizer.
- Uses no exploration noise