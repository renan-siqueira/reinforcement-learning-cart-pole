import math
import numpy as np
import gym


class QLearningAgent:
    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros(n_states + (self.n_actions,))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        return np.argmax(self.q_table[*state, :])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[*state, action]
        target = reward + self.gamma * np.max(self.q_table[*next_state, :])
        self.q_table[*state, action] = predict + self.alpha * (target - predict)


def discretize_state(env, buckets, obs):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    discrete_state = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    return tuple(min(buckets[i]-1, max(0, discrete_state[i])) for i in range(len(buckets)))


def training(env, q_agent, buckets, episodes, debug=False):
    if debug:
        log_file = open("default_env_training.log", "w")

    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretize_state(env, buckets, obs)
        step = 0
        done = False
        while not done:
            action = q_agent.choose_action(state)
            
            if debug:
                # Monitor ε-greedy choice
                if np.random.uniform(0, 1) < q_agent.epsilon:
                    log_message = f"Episode {episode+1}, Step {step}: Exploring with action {action}\n"
                else:
                    log_message = f"Episode {episode+1}, Step {step}: Exploiting with action {action}\n"
                
                log_file.write(log_message)

            next_obs, reward, done, _, _ = env.step(action)
            next_state = discretize_state(env, buckets, next_obs)
            
            if debug:
                # Monitor Q-Table Updates
                old_value = q_agent.q_table[*state, action]
                q_agent.learn(state, action, reward, next_state)
                new_value = q_agent.q_table[*state, action]
                log_message = f"Episode {episode+1}, Step {step}: State: {state}, Action: {action}, Reward: {reward}, Old Q-value: {old_value}, New Q-value: {new_value}\n"
                log_file.write(log_message)
            else:
                q_agent.learn(state, action, reward, next_state)
            
            state = next_state
            step += 1

        if (episode + 1) % 100 == 0:
            log_message = f"Episode {episode+1} completed\n"
            if debug:
                log_file.write(log_message)
            else:
                print(log_message)

    if debug:
        log_file.close()


def test_agent(env, q_agent, buckets, test_episodes):
    total_reward = 0
    for _ in range(test_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            state = discretize_state(env, buckets, obs)
            action = np.argmax(q_agent.q_table[*state, :])
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

    print(f"Average reward over {test_episodes} test trials: {total_reward / test_episodes}")


def main():

    # ===Training===
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n

    buckets = (2, 2, 6, 12)
    q_agent = QLearningAgent(n_actions, n_states=buckets)

    episodes = 5

    training(env, q_agent, buckets, episodes, debug=True)
    env.close()

    # ===Test===
    test_episodes = 100
    visual_env = gym.make('CartPole-v1', render_mode='human')

    test_agent(visual_env, q_agent, buckets, test_episodes)
    visual_env.close()


if __name__ == '__main__':
    main()
