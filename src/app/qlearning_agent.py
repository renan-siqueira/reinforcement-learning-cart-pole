import math
import numpy as np


class QLearningAgent:
    def __init__(self, n_actions, n_states, alpha, gamma, epsilon):
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
