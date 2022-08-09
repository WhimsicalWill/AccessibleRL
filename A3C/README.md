# Asynchronous Advantage Actor Critic

# Conceptual Overview

Asynchronous Advantage Actor Critic (A3C) is an on-policy method that overcomes many of the shortcomings of on-policy methods by collecting data from many processes asynchronously. A large problem for RL is that data is highly correlated, so collecting unique experiences from many parallel environments helps mitigate this concern.

This implementation operates on environments with discrete action spaces, so it can be effectively thought of as a distributed version of Deep Q-Network (DQN).

A3C maintains a centralized policy and has a set amount of agents collecting data with their own local policy. After a given amount of time steps, the gradients of the local policy are calculated and sent back to the centralized policy to update it. Then, the local policy grabs a copy of the new centralized policy, effectively syncing with the  updates of the other local policies.

Instead of collecting transitions, the agent stores `rewards`, `values` and `log_probs` in a replay buffer. 

The loss used for A3C is a combination of the `actor loss`, `critic loss` and `entropy bonus`. The entropy bonus encourages the actor to maximize the uncertainty of its policy, which enables it to learn more effectively and not get stuck in local optima.

# Networks

- Actor is a Deep NN with one hidden layer that maps states -> actions
- Critic is a Deep NN with one hidden layer that maps (state, action) -> value

This implementation uses the same hidden layer to learn representations for both the Actor and Critic, which are two different output heads.

# Learning Update

During an episode taken by a local policy, we will call the learn() function every T timesteps or when the episode is terminated. We will use all of this sampled experience to change our model parameters, and then reset our local agent's replay buffer after the update.

First, we calculate the rewards-to-go by iterating backwards from the end of the episode, summing the rewards and discounting by our discount factor, gamma, as we go.

Then, we use Generalized Advantage Estimation to calculate the advantage of any particular state. This advantage is the `value of action a in state s` - `average value in state s`. Or, formally, `Q(s,a)` - `V(s)`.

Then, we maximize `actor loss`, which is the sum of the `advantages` multiplied by the `log_probs`. Remember that these `log_probs` are the logarithms of the probability that we took the action under the given policy.

For an action with a positive advantage (better than average), we would like to increase the probability of this action. For an action with a negative advantage (worse than average), we would like to decrease the probability of the action taken. This is the intuition behind the `actor_loss`.

Additionally, the `critic loss` is calculated as the Mean Squared Error (MSE) between the critic's estimates of the Q-values and the actual rewards-to-go. We would like our critic to be accurate, so we minimize this quantity.

Next, we maximize the entropy of our actor by taking the expected information content of our policy actions. Maximum entropy is reached when our actor performs uniformly random actions. This regularization ensures that the actor model doesn't become overconfident and as a result get stuck in local minima.

Finally, we reset our local agent's replay buffer and step the centralized policy's optimizer using the gradients of our combined loss function. Once the local policy sends its gradients to the centralized controller, it copies the current policy of the central controller and continues to run episodes.

# Other Information

- Local agents use independent replay buffers
- Local agents use a shared adam optimizer.
- Uses no exploration noise