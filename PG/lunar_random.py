import gym

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    
    num_episodes = 50

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, info = env.step(action)
            env.render()
            score += reward
        print(f"score for ep {episode}: {score}")

