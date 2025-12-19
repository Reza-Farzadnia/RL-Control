import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return self.max_action * x


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 network
        self.layer1 = nn.Linear(state_dim + action_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

        # Q2 network
        self.layer4 = nn.Linear(state_dim + action_dim, 64)
        self.layer5 = nn.Linear(64, 64)
        self.layer6 = nn.Linear(64, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        # Q1
        q1 = torch.relu(self.layer1(sa))
        q1 = torch.relu(self.layer2(q1))
        q1 = self.layer3(q1)

        # Q2
        q2 = torch.relu(self.layer4(sa))
        q2 = torch.relu(self.layer5(q2))
        q2 = self.layer6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = torch.relu(self.layer1(sa))
        q1 = torch.relu(self.layer2(q1))
        q1 = self.layer3(q1)
        return q1


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in indices:
            s, a, r, s_, d = self.buffer[i]
            states.append(np.asarray(s))
            actions.append(np.asarray(a))
            rewards.append(np.asarray(r))
            next_states.append(np.asarray(s_))
            dones.append(np.asarray(d))

        return (np.array(states), np.array(actions), np.array(rewards).reshape(-1, 1),
                np.array(next_states), np.array(dones).reshape(-1, 1))

    def size(self):
        return len(self.buffer)


# TD3 Agent
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99,
                 tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action += np.random.normal(0, noise, size=action.shape)
        return action.clip(-self.max_action, self.max_action)

    def train(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return

        self.total_it += 1

        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        with torch.no_grad():
            # Select next action with target policy and add noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-values
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q-values
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
