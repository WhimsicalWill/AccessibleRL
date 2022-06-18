import gym
from blackjack import Agent

if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    agent = Agent()
    n_episodes = int(5e5)
    for i in range(n_episodes):
        if i % int(5e4) == 0:
            print('starting episode', i)
            # agent's V function is a dictionary (hashmap)
            print(agent.states[(21, 3, True)].value)
            print(agent.states[(4, 1, False)].value)
        observation = env.reset()
        done = False

        # play an episode
        while not done:
            action = agent.policy(observation)
            observation_, reward, done, info = env.step(action)
            agent.memory.append((observation, reward))
            observation = observation_
        agent.update_V()
    # agent's V function is a dictionary (hashmap)
    print(agent.states[(21, 3, True)].value)
    print(agent.states[(4, 1, False)].value)