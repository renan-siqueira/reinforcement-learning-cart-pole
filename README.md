# Q-Learning Agent for CartPole

This repository provides implementations of a Q-learning agent to balance a pole on a cart. The agent is implemented in two different environments:

1. The default `CartPole-v1` environment provided by OpenAI's `gym`.

2. A custom `CustomCartPole-v1` environment where external disturbances (like wind) are introduced and custom reward functions are used.

---

## Files in the Repository:

- `default_env_qlearning_agent.py`: Contains the implementation for training and testing the Q-learning agent on the default `CartPole-v1` environment.

- `custom_env_qlearning_agent.py`: Contains the implementation for training and testing the Q-learning agent on the custom `CustomCartPole-v1` environment.

---

## Requirements:

- Python 3.x
- numpy
- gym
- (optional) Pygame, if using Gym's 'human' rendering mode.

---

## How to Run:

### Training and Testing on the Default Environment:

```bash
python default_env_qlearning_agent.py
```

### Training and Testing on the Custom Environment:

```bash
python custom_env_qlearning_agent.py
```

---

## Functionality:

### Q-learning Agent:

The Q-learning agent is implemented with functionalities to:

- Choose an action based on an ε-greedy strategy.
- Learn by updating the Q-table based on the Bellman equation.

### Discretization:

Given that the state space of the CartPole environment is continuous, the states are discretized to fit into a Q-table.

### Training:

During training:

- The agent learns by interacting with the environment.
- The Q-values are updated based on the agent's interactions.
- Logging is done to monitor ε-greedy choice and Q-table updates.

### Testing:

After training, the Q-learning agent's performance is evaluated on several episodes. The agent uses the learned Q-values to select actions, and the average reward over the test trials is printed.

### Custom Environment (`CustomCartPole-v1`):

The custom environment introduces a few modifications:

- External disturbances, simulating wind, which can push the cart.
- A customized reward function which penalizes the agent based on the cart's position and the pendulum's angle.

### Debugging:

Both implementations include a `debug` flag during training. When `debug=True`, the choice of action (exploring vs exploiting) and Q-table updates are logged into separate log files (`default_env_training.log` for the default environment and `custom_env_training.log` for the custom environment).

---

## Credits:

This project was developed with the help of the OpenAI platform and based on tutorials and documentation from the OpenAI Gym library.
