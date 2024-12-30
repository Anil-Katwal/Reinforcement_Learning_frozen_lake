# FrozenLake Q-Learning

This project implements a **Q-learning algorithm** to train an agent in OpenAI Gym's `FrozenLake-v1` environment. The agent learns to navigate the slippery lake while maximizing its cumulative reward by avoiding holes and reaching the goal.

---

## Overview

The `FrozenLake` environment is a grid world where the agent must reach a goal without falling into holes. This implementation uses Q-learning, a reinforcement learning technique, to teach the agent optimal navigation strategies.

---

## Features

- **Q-learning Algorithm**: Utilizes the Bellman equation to iteratively update Q-values.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation during training.
- **Hyperparameter Tuning**: Customizable learning rate, discount rate, and epsilon decay.
- **Performance Visualization**: Plots showing epsilon decay and average rewards per 1000 episodes.

---

## Environment Setup

The environment is created using OpenAI Gym:

```python
env = gym.make("FrozenLake-v1", is_slippery=True, map_name="4x4", render_mode="human")
```

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/frozenlake-qlearning.git
   cd frozenlake-qlearning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the `frozenlake_qlearning.py` script:

```bash
python frozenlake_qlearning.py
```

During training, the agent learns the optimal policy over a specified number of episodes.

---

## Key Parameters

| Parameter        | Default Value | Description                                 |
|------------------|---------------|---------------------------------------------|
| `learning_rate`  | 0.7           | Rate at which the Q-values are updated.     |
| `discount_rate`  | 0.99          | Importance of future rewards.              |
| `epsilon`        | 1.0           | Initial exploration rate.                  |
| `epsilon_decay`  | 0.00028       | Decay rate for epsilon over episodes.      |
| `num_episodes`   | 30,000        | Total number of training episodes.         |
| `max_steps`      | 100           | Maximum steps per episode.                 |

---

## Output

1. **Epsilon Decay**: Plot showing how the exploration rate changes over time.
2. **Average Reward**: Plot of average rewards per 1000 episodes.

---

## Customization

- **Modify Hyperparameters**:
  Adjust parameters such as `learning_rate`, `discount_rate`, and `epsilon_decay` for better performance.

- **Change Environment**:
  Use a different map or environment settings by modifying the `gym.make` call.

---

## Dependencies

- `numpy`
- `matplotlib`
- `gym` (OpenAI Gym)

See `requirements.txt` for details.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- OpenAI for the `gym` library.
- Community tutorials and resources on reinforcement learning.

