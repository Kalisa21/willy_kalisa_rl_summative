import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from custom_env import LegalHelpEnv
from gymnasium.wrappers import FlattenObservation
import os
from torch.utils.tensorboard import SummaryWriter

# Simple policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.output(x)
        return self.softmax(logits)

def discount_rewards(rewards, gamma=0.99):
    discounted = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)
    return discounted

def train_reinforce(total_episodes=1000, gamma=0.99, learning_rate=1e-3, save_path="models/pg/reinforce_model.pt"):
    # Environment setup
    env = FlattenObservation(LegalHelpEnv(render_mode=None))
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir="tensorboard_logs/reinforce")

    print("🚀 Starting REINFORCE training...")

    for episode in range(total_episodes):
        log_probs = []
        rewards = []
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action_probs = policy(obs_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            obs, reward, done, truncated, info = env.step(action.item())
            rewards.append(reward)
            total_reward += reward

        # Compute discounted returns and loss
        discounted = discount_rewards(rewards, gamma)
        discounted = torch.tensor(discounted, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        loss = -torch.sum(log_probs * discounted)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("REINFORCE/TotalReward", total_reward, episode)
        print(f"Episode {episode+1}/{total_episodes} | Total Reward: {total_reward:.2f}")

    # Save trained model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(policy.state_dict(), save_path)
    print(f"💾 Saved REINFORCE model to {save_path}")
    writer.close()

if __name__ == "__main__":
    train_reinforce()
