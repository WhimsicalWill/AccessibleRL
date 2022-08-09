# Intrinsic Curiosity Module

# Conceptual Overview

The Intrinsic Curiosity Module (ICM) is a bolt-on module for reinforcement learning algorithms that helps to solve the exploration problem in RL by mathematically formalizing the idea of `curiosity`.

Instead of learning only from `extrinsic rewards`, we now learn from a combination of rewards, `intrinsic reward` + `extrinsic reward`. In general, actions for which the agent cannot predict the effect of the action on the environment state give the agent a high intrinsic reward. However, ICM also uses some subtle tricks to ensure that our agent does not become fascinated by parts of the environment that are intrinsically uncertain.

This implementation combines A3C with ICM, but other algorithms can easily be combined with ICM as well.

# Networks

- Phi is a Deep NN encoder mapping states -> state embeddings
- Inverse Model is a Deep NN with one hidden layer mapping (state_embedding, next_state_embedding) -> action distribution
- Forward Model is a Deep NN with one hidden layer mapping (state_embedding, action) -> next state embedding

# Intuition about ICM

There are two components of the ICM loss, namely the `inverse_loss` and the `forward_loss`.

For the `inverse_loss`:

1. First, the model obtains a predicted action distribution from the inverse model which takes two state embeddings as input.
2. Then, the loss is measured as the distance (KL divergence) between the predicted action distribution and the actual action distribution.

The intuition for this loss is that by predicting actions, our embedding space (phi) will capture features relevant to predicting actions. This eliminates the "noisy TV" problem. From the ICM paper: "As there is **no incentive** for φ(st) to encode any environmental features that **can not influence** or **are not influenced by** the agent’s actions, the learned exploration strategy of our agent is robust to uncontrollable aspects of the environment"

For the forward_loss:

1. Feed a state embedding and action to a neural network that predicts the resultant state embedding.
2. Then take the MSE loss between the predicted embedding and the actual embedding.

As we see more examples of a specific type of transition, we can more reliably predict the effects of our actions in the environment, and we have less `curiosity` about the given state action pair. Thus, our agent's `curiosity` may start out high but lower as a given transition becomes less 'interesting', i.e. more predictable.

The combined loss `inverse_loss + forward_loss` essentially says: predict some aspect of the resultant state embedding as well as possible, under the constraint that the embedding should contain features relevant to predicting the action. If not for this constraint, many trivial solutions would be possible, such as an encoder that maps all states to zero.

# Learning Update

There are two core ideas to the learning update of an algorithm using ICM. The first idea is that we must add the `intrinsic reward` to the `extrinsic reward` and follow the base algorithm's update rule using this combined reward. The second idea is that we must update the parameters specific to the Intrinsic Curiosity Module itself.

Calculate intrinsic reward by taking the Mean Squared Error (MSE) of the predicted state embedding with the actual state embedding, and add this to the environment reward. Once we have factored in this information about how interesting a given environment transition is, we can follow the base algorithm's update rule as usual.

To update the parameters of ICM, we take a gradient step in the direction that minimizes the combined `inverse_loss + forward_loss`

Where `inverse_loss` is the distance (KL divergence) between the predicted action distribution and the observed action distribution, and `forward_loss` is the MSE between the predicted next state embedding and the observed next state embedding.

# Other Information

- This code is an implementation of A3C + ICM, but it can be combined with any algo
- Since we build on top of A3C, we have a global ICM that aggregates the gradient steps of local ICMs that run in parallel