import numpy as np
from gym.envs.classic_control import CartPoleEnv


class CustomCartPole(CartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(CustomCartPole, self).__init__(*args, **kwargs)
        self.masspole *= 1.0 # mast weight
        self.length *= 1.0 # mast length
        self.wind = False

    def step(self, action):
        observation, reward, done, _, _ = super(CustomCartPole, self).step(action)

        if self.wind:
            # Introduce external disturbances (wind)
            wind = np.random.uniform(-0.05, 0.05)  # simulates a random wind force
            observation[0] += wind

        # Customizing the reward
        x_position = observation[0]   # cart position
        pole_angle = observation[2]   # pendulum angle

        # Penalizes the agent as he moves away from the center
        reward -= abs(x_position) * 0.5

        # Penalizes the agent as the pendulum swings
        reward -= abs(pole_angle) * 2

        return observation, reward, done, False, {}
