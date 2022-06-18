import gym
import numpy as np



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    num_eps = 5000
    action_space = np.array([0, 1])
    pole_velocities = []
    cart_velocities = []

      
    # use random agent to get general bounds for velocities
    for episode in range(num_eps):
        done = False
        observation = env.reset()
        while not done:
            observation_, _, done, _ = env.step(np.random.choice(action_space))
            _, cart_vel, _, pole_vel = observation_
            pole_velocities.append(pole_vel)
            cart_velocities.append(cart_vel)
    
    print(f"pole_vel: min={min(pole_velocities)}, avg={np.mean(pole_velocities)}, max={max(pole_velocities)}") 
    print(f"cart_vel: min={min(cart_velocities)}, avg={np.mean(cart_velocities)}, max={max(cart_velocities)}") 

# bounds are roughly [-3.6, 3.6] for both velocity intervals

