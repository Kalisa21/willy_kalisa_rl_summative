import torch
import numpy as np
import time
from custom_env import LegalHelpEnv
from gymnasium.wrappers import FlattenObservation

# Same policy architecture used in training
class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.output = torch.nn.Linear(64, output_dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.softmax(self.output(x))

def evaluate_reinforce(model_path="models/pg/reinforce_model.pt", episodes=5, render=True):
    env = FlattenObservation(LegalHelpEnv(render_mode="human" if render else None))
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load policy
    policy = PolicyNetwork(obs_dim, action_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"\n🎬 REINFORCE Episode {ep + 1}")

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action_probs = policy(obs_tensor)
            action = torch.argmax(action_probs).item()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if render:
                env.render()
                time.sleep(0.3)

        print(f"🏁 Episode {ep + 1} finished in {step} steps | Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate_reinforce()
