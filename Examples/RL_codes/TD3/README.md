# TD3 Toy Example

A simple, clean implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3) for continuous state and action spaces.

## Files

- **td3_agent.py**: TD3 agent module containing:
  - `Actor`: Neural network for policy
  - `Critic`: Twin Q-networks for value estimation
  - `ReplayBuffer`: Experience replay storage
  - `TD3Agent`: Main agent with training logic

- **ToyExample_TD3.py**: Main training script with:
  - `DampedOscillator`: Simple continuous environment
  - Training loop
  - Testing/evaluation

## Environment

**Damped Oscillator**: A mass that needs to reach the origin (position=0, velocity=0) by applying continuous force.

- **State**: [position, velocity]
- **Action**: [force] (continuous)
- **Reward**: Negative distance to goal
- **Goal**: Reach close to origin

## How to Run

```bash
python ToyExample_TD3.py
```

## Key TD3 Features

1. **Twin Critics**: Two Q-networks to reduce overestimation
2. **Delayed Policy Updates**: Actor updated less frequently than critics
3. **Target Networks**: Smoothly updated for stability
4. **Exploration Noise**: Added during training for exploration

## Requirements

- numpy
- torch (PyTorch)
