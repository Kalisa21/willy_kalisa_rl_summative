# reinforce_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from custom_env import LegalHelpEnv
import numpy as np
import os

# Set up logging folders
os.makedirs("models/reinforce", exist_ok=True)

# Hyperparameters
learning_rate = 1e-3
gamma = 0.99
num_episodes = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple MLP Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# REINFORCE Training Function
def train():
    env = LegalHelpEnv()
    obs_space = env.observation_space["agent"].shape[0]
    act_space = env.action_space.n

    policy = PolicyNetwork(obs_space, act_space).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        log_probs = []
        rewards = []

        while not done:
            state = torch.tensor(obs["agent"], dtype=torch.float32).to(device)
            probs = policy(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        loss = []
        for log_prob, G in zip(log_probs, returns):
            loss.append(-log_prob * G)
        loss = torch.stack(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode+1}/{num_episodes} | Total Reward: {sum(rewards):.2f}")

    torch.save(policy.state_dict(), "models/reinforce/wombguard_reinforce_model.pt")
    print("✅ REINFORCE training complete.")

if __name__ == "__main__":
    train()
