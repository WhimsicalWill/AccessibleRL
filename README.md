# Accessible RL

# Features

- Pytorch implementations of 8 RL algos
- Conceptual overviews
- Easy to run experiments
- Configurable hyperparams

# Installation

The easiest way to get started is to create a virtual environment and install the required dependencies with pip


```bash
  git clone https://github.com/WhimsicalWill/ActorCritic.git
  cd ActorCritic
  pip install -r requirements.txt
```

# Code Structure

The goal of this repository is to give clear and consistent implementations of different RL algorithms by breaking them down into their constituent parts. Each algorithm uses the same abstraction of an agent who takes actions, stores experience, and learns from it. While this agent abstraction is consistent, pay attention to the differences in `networks.py` and the `learn` function of each agent. This is where the algorithms differ, and where you can grow and reinforce your understanding of each algo. 

# Documentation

Each algorithm has an accompanying `README.md` with an explanation of the algorithm, its specific implementation, and its hyperparameters.

# Running Experiments

A great way to learn about existing algorithms is to modify their code, tweak their hyperparameters, and to break them (which is very easy to do). This repo is structured to allow for experiments to be run easily.

```bash
  python {algo_name}/train.py
```

# TODO:

- Write `README.md` for each algo
- Create minimal `environment.yaml`
- Put hyperparams in config.json