# Q-Learning Agent for CartPole

This repository provides implementations of a Q-learning agent to balance a pole on a cart. The agent is implemented in two different environments:

1. The default `CartPole-v1` environment provided by OpenAI's `gym`.

2. A custom `CustomCartPole-v1` environment where external disturbances (like wind) are introduced and custom reward functions are used.

---

## Requirements:

- Python 3.x
- numpy
- gym
- (optional) Pygame, if using Gym's 'human' rendering mode.

To install the required libraries, you can use:

```bash
pip install -r requirements.txt
```

---

## Directory Structure:

.
|-- run.py                     # Main execution script
|-- src/
|   |-- app/
|   |   |-- custom_env.py      # Custom environment definition
|   |   `-- qlearning_agent.py # Q-learning agent definition
|   |-- modules/
|   |   |-- testing.py         # Agent testing procedures
|   |   `-- training.py        # Agent training procedures
`-- log/
    `-- training.log           # Log file for debugging during training

---

## How to Run:

### Training and Testing on the Default Environment:

```bash
python run.py
```

### Training and Testing on the Custom Environment:

```bash
python run.py --custom
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

Both implementations feature a `debug` option during training. If set to `debug=True`, the log captures the nature of action chosen (exploration vs. exploitation) and any changes to the Q-table. The log can be found at `log/training.log`.

---

## Acknowledgments:

This venture was realized with the support of the OpenAI platform, and it draws inspiration from tutorials and resources available on the OpenAI Gym library.
