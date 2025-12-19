import numpy as np
from td3_agent import TD3Agent

# Simple Continuous Environment: Damped Oscillator
# Goal: Move a mass to the origin (position = 0, velocity = 0) using continuous force
class DampedOscillator:
    def __init__(self):
        self.dt = 0.1  # time step
        self.mass = 1.0
        self.damping = 0.1
        self.max_force = 2.0
        self.max_position = 5.0
        self.max_velocity = 5.0
        self.state = None

    def reset(self):
        # Random initial position and velocity
        self.state = np.array([
            np.random.uniform(-2.0, 2.0),  # position
            np.random.uniform(-1.0, 1.0)   # velocity
        ])
        return self.state

    def step(self, action):
        # Extract state
        position, velocity = self.state
        force = np.clip(action[0], -self.max_force, self.max_force)

        # Physics: F = ma, with damping
        acceleration = (force - self.damping * velocity) / self.mass

        # Update state (simple Euler integration)
        new_velocity = velocity + acceleration * self.dt
        new_position = position + new_velocity * self.dt

        # Clip to bounds
        new_position = np.clip(new_position, -self.max_position, self.max_position)
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)

        self.state = np.array([new_position, new_velocity])

        # Reward: negative distance to goal (origin)
        distance_to_goal = np.sqrt(new_position**2 + new_velocity**2)
        reward = -distance_to_goal

        # Episode ends if close to goal or too far
        done = distance_to_goal < 0.1 or abs(new_position) > self.max_position - 0.1

        return self.state, reward, done


# Training
env = DampedOscillator()
state_dim = 2  # [position, velocity]
action_dim = 1  # [force]
max_action = env.max_force

agent = TD3Agent(state_dim, action_dim, max_action)

episodes = 200
max_steps = 100

print("Training TD3 agent...")
for ep in range(episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # Select action with exploration noise
        action = agent.select_action(state, noise=0.1)

        # Take action in environment
        next_state, reward, done = env.step(action)
        episode_reward += reward

        # Store transition in replay buffer
        agent.replay_buffer.add(state, action, reward, next_state, done)

        # Train agent
        agent.train(batch_size=64)

        state = next_state

        if done:
            break

    if (ep + 1) % 20 == 0:
        print(f"Episode {ep + 1}/{episodes}, Reward: {episode_reward:.2f}")

print("\nTraining complete!")

# Test the trained agent
print("\n" + "="*50)
print("Testing trained agent (5 episodes):")
print("="*50)

for test_ep in range(5):
    state = env.reset()
    print(f"\nTest Episode {test_ep + 1}:")
    print(f"  Initial state: pos={state[0]:.3f}, vel={state[1]:.3f}")

    episode_reward = 0
    trajectory = [state.copy()]

    for step in range(max_steps):
        # Select action without noise
        action = agent.select_action(state, noise=0.0)
        next_state, reward, done = env.step(action)

        episode_reward += reward
        trajectory.append(next_state.copy())
        state = next_state

        if done:
            break

    final_state = trajectory[-1]
    print(f"  Final state: pos={final_state[0]:.3f}, vel={final_state[1]:.3f}")
    print(f"  Steps taken: {len(trajectory) - 1}")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Distance to goal: {np.sqrt(final_state[0]**2 + final_state[1]**2):.3f}")
