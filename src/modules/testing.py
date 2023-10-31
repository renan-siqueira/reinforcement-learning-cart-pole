import numpy as np
from ..app.qlearning_agent import discretize_state


def test_agent(env, q_agent, buckets, test_episodes):
    total_reward = 0
    for episode in range(test_episodes):
        obs, _ = env.reset()
        done = False
        print(f"Testing on Episode: {episode + 1}")
        while not done:
            state = discretize_state(env, buckets, obs)
            action = np.argmax(q_agent.q_table[*state, :])
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

    print(f"Average reward over {test_episodes} test trials: {total_reward / test_episodes}")
