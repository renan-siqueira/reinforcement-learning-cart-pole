import argparse
import gym
from gym.envs.registration import register

from src.app.custom_env import CustomCartPole
from src.app.qlearning_agent import QLearningAgent
from src.modules.training import training
from src.modules.testing import test_agent


def main(env_name, buckets, training_episodes, test_episodes, agent_params, debug):

    if env_name == 'CustomCartPole-v1':
        register(
            id='CustomCartPole-v1',
            entry_point='__main__:CustomCartPole',
            max_episode_steps=500,
            reward_threshold=475.0,
        )

    env = gym.make(env_name)
    n_actions = env.action_space.n
    q_agent = QLearningAgent(n_actions, n_states=buckets, **agent_params)

    training(env, q_agent, buckets, training_episodes, debug=debug)
    env.close()

    visual_env = gym.make(env_name, render_mode='human')
    test_agent(visual_env, q_agent, buckets, test_episodes)
    visual_env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Choose the environment for the Q-learning agent.")
    parser.add_argument('--custom', action='store_true', help="Use the custom environment.")
    args = parser.parse_args()

    buckets = (2, 2, 6, 12)
    training_episodes = 100
    test_episodes = 3

    agent_params = {
        'alpha': 0.1,
        'gamma': 0.5,
        'epsilon': 0.6
    }

    # Use 'CartPole-v1' for default and 'CustomCartPole-v1' for --custom
    env_choice = 'CustomCartPole-v1' if args.custom else 'CartPole-v1'

    main(env_choice, buckets, training_episodes, test_episodes, agent_params, debug=True)
