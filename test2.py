import numpy as np
import gym


RENDER = False
MONITOR = False

class RandomGuessingAgent():
    def __init__(self, W):
        self.W = W

    def action(self, state):
        if np.dot(self.W, state) > 0:
            return 1
        else:
            return 0


def test_agent(env, agent, n_steps=1000, n_episodes=100):
    agent_reward = 0
    for _ in range(n_episodes):
        episode_reward = run_episode(env, agent, n_steps)
        agent_reward += episode_reward
    return agent_reward


def run_episode(env, agent, n_steps=1000):
    state = env.reset()
    episode_reward = 0
    for t in range(n_steps):
        if RENDER:
            env.render()
        state, reward, done, info = env.step(
            agent.action(np.array(state)))
        episode_reward += reward
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    return episode_reward


def train_random_guessing(n_guesses=10000, n_episodes=100, n_steps=100):
    env = gym.make('CartPole-v0')
    if MONITOR:
        env.monitor.start('trials/1', force=True)

    best_agent = None
    best_reward = None
    agent_rewards = []
    for _ in range(n_guesses):
        agent = RandomGuessingAgent(np.random.random((1, len(env.state))))
        agent_reward = test_agent(env, agent, n_steps=n_steps, n_episodes=n_episodes)

        print 'Agent reward', agent_reward
        agent_reward /= float(n_episodes * n_steps)
        print 'Agent reward norm', agent_reward
        if agent_reward > 0.8:
            print agent.W
        agent_rewards.append(agent_reward)
        if best_reward is None or agent_reward > best_reward:
            best_reward = agent_reward
            best_agent = agent

    if MONITOR:
        env.monitor.close()
    print agent_rewards


agent = RandomGuessingAgent(
    np.array([0.27331873, 0.55931623, 0.93337996, 0.82882202])
)

env = gym.make('CartPole-v0')
env.monitor.start('trials/1', force=True)
print test_agent(env, agent, n_steps=1000, n_episodes=100)
env.monitor.close()

#train_random_guessing(n_guesses=10, n_episodes=100, n_steps=10000)    
