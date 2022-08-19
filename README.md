# Accessible RL

This repo is intended as a learning resource to supplement theoretical study of reinforcement learning with actual implementation details of some of the most important RL algorithms.

# Features

- PyTorch implementations of 8 RL algos
- Conceptual overviews
- Simple installation
- Easily run experiments
- Configurable hyperparams

# Code Structure

The goal of this repository is to give clear and consistent implementations of different RL algorithms by breaking them down into their constituent parts. Each algorithm uses the same abstraction of an agent who takes actions, stores experience, and learns from it. While this agent abstraction is consistent, pay attention to the differences in `networks.py` and the `learn` function of each agent. This is where the algorithms differ, and where you can grow and reinforce your understanding of each algo. 

# Installation

The easiest way to get started is to create a virtual environment and install the required dependencies with pip

```bash
  git clone https://github.com/WhimsicalWill/ActorCritic.git
  cd ActorCritic
  pip install -r requirements.txt
```

# Documentation

Each algorithm has an accompanying `README.md` with an explanation of the algorithm, its specific implementation, and its hyperparameters.

# Running Experiments

A great way to learn about existing algorithms is to modify their code, tweak their hyperparameters, and to break them (which is very easy to do). This repo is structured to allow for experiments to be run easily.

Train an RL agent on an environment with a specific algo

```bash
  python {algo_name}/main.py -e "{env_name}"
```

Render sample games from the trained agent on the environment it was trained on

```bash
  python {algo_name}/main.py -e "{env_name}" -r
```

# Suggested Curriculum

1. PG
2. DQN
3. DDPG
4. TD3
5. SAC
6. A3C
7. ICM
8. PPO

# TODO:

- Fix structure of A3C and ICM (similar)