# Intrinsic Curiosity Module

# Conceptual Overview

The Intrinsic Curiosity Module (ICM) is a bolt-on module for reinforcement learning algorithms that helps to solve the exploration problem in RL by mathematically formalizing the ideas of intrinsic `motivation` and `curiosity`.

Insetad of merely learning from `extrinsic rewards`, we now learn from a combination of rewards, `intrinsic reward` + `extrinsic reward`. In general, actions for which the agent cannot predict the effect of the action on the environment state give the agent a high intrinsic reward. However, ICM uses some subtle tricks to ensure that our agent does not become fascinated by parts of the environment that are intrinsically uncertain.

This implementation combines A3C with ICM, but other algorithms can easily be combined with ICM as well.

# Networks
- Phi is a Deep NN encoder mapping states -> state embeddings
- Inverse Model is a Deep NN with one hidden layer mapping (state_embedding, next_state_embedding) -> action distribution
- Forward Model is a Deep NN with one hidden layer mapping (state_embedding, action) -> s'

# Intuition about ICM

There are two components of the ICM loss, namely the `inverse_loss` and the `forward_loss`.

For the `inverse_loss`:

1. First, the model obtains the action prediction from the inverse model which takes two state embeddings as input.
2. Then, the loss is measured as the distance (KL divergence) between the predicted action distribution and the actual action distribution.

The intuition for this loss is that by predicting actions, our embedding space (phi) will capture features relevant to predicting actions. This eliminates the "noisy TV" problem.

For the forward_loss:

1. Feed a state embedding and action to a neural network that predicts the resultant state embedding.
2. Then take the MSE loss between the predicted embedding and the actual embedding.

Since our encoder phi has features that obtain information relevant to predicting the action, we combine an embedding of a state and an action, and feed this to the forward model.

If our embedding represented the entire state space, it may encode features irrelevant to the action we took, e.g. "leaves blowing in the wind". Constraining it to action-related features allows us to focus on the consequences of our actions in the world.

In summary:

The combined loss `inverse_loss + forward_loss` essentially says: predict some aspect of the resultant state embedding as well as possible, under the constraint that the embedding should contain feaatures relevant to predicting the action.

If not for this constraint, many trivial solutions would be possible, such as an encoder that maps all states to zero.

From the ICM paper: "As there is **no incentive** for φ(st) to encode any environmental features that **can not influence** or **are not influenced by** the agent’s actions, the learned exploration strategy of our agent is robust to uncontrollable aspects of the environment"