import numpy as np
import matplotlib.pyplot as plt
import gym
from control_cartpole import Agent

class CartPoleStateDigitizer():
    def __init__(self, bounds=(2.4, 4, 0.209, 4), n_bins=10)